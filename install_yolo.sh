#!/bin/bash

# Installation script for YOLO v8 integration with Spark robot
# This script installs the necessary dependencies for running YOLO v8 on the robot

echo "Installing YOLO v8 dependencies for Spark robot..."

# Update package list
sudo apt update

# Install Python 3 and pip if not already installed
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment for YOLO dependencies
echo "Creating virtual environment..."
python3 -m venv ~/yolo_env
source ~/yolo_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for compatibility)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install YOLO v8
echo "Installing YOLO v8..."
pip install ultralytics

# Install other dependencies
echo "Installing additional dependencies..."
pip install opencv-python numpy Pillow

# Ensure numpy compatibility with ROS
pip install "numpy<2.0"

# Download YOLO model if not present
MODEL_PATH="./yolo11n.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading YOLO11n model..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
print('YOLO model downloaded successfully')
"
else
    echo "YOLO model already exists at $MODEL_PATH"
fi

# Make Python scripts executable
chmod +x yolo_detector.py
chmod +x grasp_yolo.py

echo "Installation completed!"
echo ""
echo "To use YOLO detection:"
echo "1. Source the virtual environment: source ~/yolo_env/bin/activate"
echo "2. Launch with YOLO: roslaunch semi_auto_match_24 semi_auto_yolo.launch use_yolo:=true"
echo "3. Or launch with original color detection: roslaunch semi_auto_match_24 semi_auto_yolo.launch use_yolo:=false"
echo ""
echo "Note: Make sure your camera is properly calibrated and the calibration file exists at ~/thefile.txt"
