# BFRB Real-Time Recognition UI

A desktop application for real-time recognition of Body-Focused Repetitive Behaviors (BFRB) using computer vision and machine learning.

## Features

- Real-time video processing with webcam
- Holistic pose and hand landmark detection using MediaPipe
- PyTorch-based classification model
- PyQt5 GUI for user interface

## Requirements

- Python 3.11
- Webcam access

## Installation

1. Clone or download the project files.

2. Create a virtual environment:

   ```bash
   py -3.11 -m venv bfrb_env_311
   ```

3. Activate the virtual environment:

   ```bash
   bfrb_env_311\Scripts\activate
   ```

4. Install the required packages:

   ```bash
   pip install PyQt5 opencv-python mediapipe==0.10.9 torch torchvision scikit-learn numpy pandas openpyxl
   ```

## Usage

1. Ensure `model_state_dict.pt` and `model_class.py` are in the same directory as `bfrb_realtime_ui.py`.

2. Run the application:

   ```bash
   python bfrb_realtime_ui.py
   ```

3. The GUI will open. Once the GUI opens, upload the model file (model_state_dict.pt) if prompted.

4. Start real-time recognition using the interface.

## Files

- `bfrb_realtime_ui.py`: Main application script
- `model_class.py`: PyTorch model definition
- `model_state_dict.pt`: Trained model weights
- `bfrb_env_311/`: Virtual environment directory

## Notes

- The application uses MediaPipe with model_complexity=2 and refine_face_landmarks for accurate landmark detection.
- Feature extraction mirrors the training pipeline for consistent results.