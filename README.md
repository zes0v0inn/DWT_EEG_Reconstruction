# The official code for the Paper *Category-aware EEG Image Generation Based on Wavelet Transform and Contrast Semantic Loss*
The paper has been accepted by IJCAI 2025.

# Getting Started
## Prerequisites
- Python 3.8 or higher
- PyTorch 1.10+
- NumPy, SciPy, Matplotlib
- Additional dependencies listed in requirements.txt

# Installation
## Clone the repository:
```bash
git clone https://github.com/zes0v0inn/DWT_EEG_Reconstruction.git
cd DWT_EEG_Reconstruction
```
## Install the required packages:
```bash
pip install -r requirements.txt
# Some additional libraries need to be downloaded separately.
pip install wandb einops open_clip_torch transformers==4.28.0.dev0 diffusers==0.24.0 braindecode==0.8.1
```

# Data Preparation
Ensure that your EEG dataset is organized according to the structure expected by data_config.json. Update the configuration file with the correct paths to your data.
The official dataset can be found in https://osf.io/anp5v/wiki/home/ .

# Training
To train the model, run:
```bash
python DWTReconstruction_Training.py --config DWTreconstruction.yml
```
The detailed arguments can be found in the python file. You can freely change the detailed parameters for you own experiments.

# EEG Image Generation
Use the provided Jupyter notebook `new_gen.ipynb` to generate and visualize reconstructed EEG images.

# Acknowledgement
We thank the authors and the lab members for their support and valuable feedback.
We also want to exhibit our great appreciations towards the authors of [ATMS](https://github.com/dongyangli-del/EEG_Image_decode?tab=readme-ov-file), [NICE](https://github.com/eeyhsong/NICE-EEG), [Necomimi](https://arxiv.org/abs/2410.00712).
