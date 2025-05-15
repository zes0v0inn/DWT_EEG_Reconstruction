import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
import itertools
import math
import re
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
import numpy as np
from loss import ClipLoss, FeatureContrastLoss
import argparse
from torch import nn
from torch.optim import AdamW
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class WaveBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(WaveBlock, self).__init__()
        self.dwt = DWT1DForward(J=1, wave='db1', mode='zero')
        self.dwt.to('cuda')
        self.conv = nn.Conv1d(c_in, c_out, 3, 1, padding=1)
    def forward(self, x):

        xl, xh = self.dwt(x)

        x2 = torch.cat((xl, xh[0]), dim=-1)
        x3 = self.conv(x2)
        return x3

class iWaveBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(iWaveBlock, self).__init__()

        self.idwt = DWT1DInverse(wave='db1', mode='zero')
        self.idwt.to('cuda')

    def forward(self, x ):
        xl, xh = torch.split(x, [128, 128], dim=-1)
        x4 = self.idwt([xl, [xh]])
        return x4


class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250  # Sequence length
        self.pred_len = 256  # Prediction length
        self.output_attention = False  # Whether to output attention weights
        self.d_model = 256  # Model dimension
        self.embed = 'timeF'  # Time encoding method
        self.freq = 'h'  # Time frequency
        self.dropout = 0.25  # Dropout rate
        self.factor = 1  # Attention scaling factor
        self.n_heads = 4  # Number of attention heads
        self.e_layers = 1  # Number of encoder layers
        self.d_ff = 256  # Feedforward network dimension
        self.activation = 'gelu'  # Activation function
        self.enc_in = 63  # Encoder input dimension (example value)


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, joint_train=False, num_subjects=num_subjects)

        self.dwt = WaveBlock(64, 64)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.local_conv_time = nn.Conv1d(63, 63, kernel_size=3, padding=3 // 2,
                                         groups=63)
        # Cross-electrode convolution
        self.local_conv_electrode = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(3 // 2, 0),
                                              groups=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(63, 63, 1),
            nn.Sigmoid()
        )

        self.idwt = iWaveBlock(63, 63)
        self.conv = nn.Conv1d(63, 63, 4, 2, padding=1)
    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        a = self.dwt(enc_out)
        a = a[:, :63, :]
        local_time = self.local_conv_time(a)
        local_time_expanded = local_time.unsqueeze(1)  # (Batch, 1, Channels, Signal_Length)
        local_electrode = self.local_conv_electrode(local_time_expanded).squeeze(1)  # (Batch, Channels, Signal_Length)
        out, attns = self.encoder(a, attn_mask=None)
        out = out[:, :63, :]
        weight = self.gate(a)  # (Batch, Channels, 1)
        out = local_electrode * (1 - weight) + out * weight
        b = self.idwt(out)
        c = self.conv(torch.cat([out, b], dim=-1))
        #print("c", c.shape)
        return c


class PrintShape(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name} shape: {x.shape}")
        return x


class MultiBranchNet(nn.Module):
    def __init__(self,
                 in_channels=63,
                 n_filters1=40,
                 n_filters2=40,
                 dropout_rate=0.5):
        super().__init__()

        self.temporal_branch = nn.Sequential(
            nn.Conv2d(in_channels, n_filters1, kernel_size=(1, 25), padding='same'),
            nn.BatchNorm2d(n_filters1),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(n_filters1, n_filters1, kernel_size=(1, 1)),
            nn.BatchNorm2d(n_filters1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 37))
        )

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, n_filters1, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(n_filters1),
            nn.ELU(),
            nn.Conv2d(n_filters1, n_filters1 * 2, kernel_size=(3, 3),
                      padding='same', groups=n_filters1),
            nn.BatchNorm2d(n_filters1 * 2),
            nn.ELU(),
            nn.Conv2d(n_filters1 * 2, n_filters1, kernel_size=1),
            nn.BatchNorm2d(n_filters1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 37))
        )

        self.fusion_branch = nn.Sequential(
            nn.Conv2d(n_filters1 * 2, n_filters2, kernel_size=1),
            nn.BatchNorm2d(n_filters2),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool2d((1, 37))
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_filters1 * 2, n_filters1 // 2, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(n_filters1 // 2, n_filters1 * 2, kernel_size=1),
            nn.Sigmoid()
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        temporal_features = self.temporal_branch(x)
        spatial_features = self.spatial_branch(x)
        combined_features = torch.cat([temporal_features, spatial_features], dim=1)
        attention_weights = self.attention(combined_features)
        weighted_features = combined_features * attention_weights
        output = self.fusion_branch(weighted_features)
        output = self.projection(output)

        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding='same', groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            MultiBranchNet(),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1480, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATMDWT(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=2048,
                 num_blocks=1):
        super(ATMDWT, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
        self.loss2 = FeatureContrastLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        # print(f'After attention shape: {x.shape}')
        # print("x", x.shape)
        # x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)

        out = self.proj_eeg(eeg_embedding)
        return out

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


def train_model(sub, eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, config):
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()  # (n_cls, d)
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.90
    beta = 0.9
    features_list = []  # List to store features
    save_features = True
    mse_loss_fn = nn.MSELoss()
    loss_CL = FeatureContrastLoss()
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()

        batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
        subject_id = extract_id_from_string(sub)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
        eeg_features = eeg_model(eeg_data, subject_ids).float()

        features_list.append(eeg_features)
        logit_scale = eeg_model.logit_scale

        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        regress_loss = mse_loss_fn(eeg_features, img_features)
        loss = beta * (alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10) + (1 - beta) * loss_CL(eeg_features, labels)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1)  # (n_batch, ) ∈ {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)


def evaluate_model(sub, eeg_model, dataloader, device, text_features_all, img_features_all, k, config):
    eeg_model.eval()

    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    beta = 0.9
    top5_correct = 0
    top5_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    loss_CL = FeatureContrastLoss()
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            eeg_features = eeg_model(eeg_data, subject_ids)
            logit_scale = eeg_model.logit_scale
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            regress_loss = mse_loss_fn(eeg_features, img_features)
            loss = beta * (alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10) + (1 - beta) * loss_CL(eeg_features, labels)
            total_loss += loss.item()

            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                selected_text_features = text_features_all[selected_classes]
                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                logits_single = logits_img
                # print("logits_single", logits_single.shape)
                # Get predicted class
                # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                predicted_label = selected_classes[
                    torch.argmax(logits_single).item()]  # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
                if predicted_label == label.item():
                    # print("predicted_label", predicted_label)
                    correct += 1
                # logits_single is the model output, assumed to be shape (n_batch, n_classes)
                # label is the true label, shape (n_batch,)
                # Get top-5 predicted indices
                # print("logits_single", logits_single)
                _, top5_indices = torch.topk(logits_single, 5, largest=True)
                # Check if true label is in top-5 predictions
                if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                    top5_correct_count += 1
                total += 1
                
            del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc


def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device,
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all,
                    config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model, logger)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []


    best_accuracy = 0.0
    results = []  # List to store results for each epoch

    for epoch in range(config.epochs):
        train_loss, train_accuracy, features_tensor = train_model(sub, eeg_model, train_dataloader, optimizer, device,
                                                                  text_features_train_all, img_features_train_all,
                                                                  config=config)
        if (epoch + 1) % 1 == 0:
            # Save the model every epoch
            if config.insubject == True:
                os.makedirs(f"./models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)
                file_path = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch + 1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            else:
                os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)
                file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/{epoch + 1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"Model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model
        test_loss, test_accuracy, top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device,
                                                            text_features_test_all, img_features_test_all, k=200,
                                                            config=config)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Append results for this epoch
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "top5_acc": top5_acc,
        }

        results.append(epoch_results)
        # If the test accuracy of the current epoch is the best, save the model and related information
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()

            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "Epoch": epoch
        })

        print(
            f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
    logger.finish()
    return results


import datetime


def main():
    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str,
                        default="/media/hanakawalab/Transcend/THINGS-Data/EEG/Preprocessed_data_250Hz",
                        help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--name', type=str, default="DWTReconstruction_{23}", help='Experiment name')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Device to run on (cpu or gpu)')
    parser.add_argument('--encoder_type', type=str, default='ATMDWT', help='Encoder type')
    parser.add_argument('--subjects', nargs='+',
                        default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08',
                                 'sub-09', 'sub-10'], help='List of subject IDs')
    args = parser.parse_args()

    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    subjects = args.subjects
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in subjects:
        eeg_model = globals()[args.encoder_type]()
        eeg_model.to(device)

        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=args.lr)

        train_dataset = EEGDataset(args.data_path, subjects=[sub], train=True)
        test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, current_time, eeg_model, train_loader, test_loader, optimizer, device,
                                  text_features_train_all, text_features_test_all, img_features_train_all,
                                  img_features_test_all, config=args, logger=args.logger)

        # Save results to a CSV file
        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)

        results_file = f"{results_dir}/{args.encoder_type}_{sub}.csv"


        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')


if __name__ == '__main__':
    main()
