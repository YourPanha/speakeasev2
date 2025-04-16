# Sign Language to Text and Speech Conversion

This project converts sign language gestures into text and speech output using computer vision and deep learning techniques.

## Features
- Real-time sign language recognition
- Support for both ASL (American Sign Language) and ISL (Indian Sign Language)
- Text output of recognized signs
- Speech synthesis of recognized signs
- Web interface for easy interaction
- Support for A-Z alphabet signs

## Prerequisites
- Python 3.8 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Collection for ISL

Before using the Indian Sign Language feature, you need to collect data for training:

1. Run the ISL data collection script:
```bash
python data_collection_isl.py
```

2. Follow the on-screen instructions to collect hand sign data for each letter
   - Press 'n' to move to the next letter
   - Press 'a' to start/stop automatic capture
   - Press 'c' to capture a single frame
   - Press 'Esc' to exit

3. Train the ISL model using the collected data:
```bash
python train_isl_model.py
```

## Usage

4. Run the main application:
```bash
python final_pred.py
```

5. Use the interface to switch between ASL and ISL recognition modes

## Key Features
- Toggle between ASL and ISL recognition
- Real-time hand tracking and gesture recognition
- Text display of recognized signs
- Speech output of recognized text
- Word suggestions based on current input

## Troubleshooting

If you encounter issues with the webcam or model loading:
1. Make sure your webcam is properly connected and accessible
2. Check that all dependencies are correctly installed
3. Verify that the model paths in the code match your actual file locations

## License
This project is provided as open-source software.

## Acknowledgements
- This project uses TensorFlow and OpenCV for machine learning and computer vision
- Hand tracking implemented using the CVZone library
- Text-to-speech conversion using pyttsx3


