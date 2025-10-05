# Steganography with Graph Neural Networks

A deep learning approach to image steganography using Graph Neural Networks (GNNs) for hiding and revealing secret images within cover images.

## Features

- **Graph-based Architecture**: Utilizes dynamic graph convolutions with window-based attention
- **High Capacity**: Capable of hiding full-resolution secret images
- **Quality Preservation**: Maintains visual quality of cover images
- **End-to-End Training**: Joint optimization of hiding and revealing processes

## Training setup

# Step 1: Create a conda environment
conda create --name stegGNN 

# Step 2: Activate the environment
conda activate stegGNN

# Step 3: Install dependencies from requirements.txt
pip install -r requirements.txt

# Step 4: Run the training script
python scripts/train.py

