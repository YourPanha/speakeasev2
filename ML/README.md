# Sign Language to Text & Speech Conversion

## 📌 Introduction
This project aims to bridge the communication gap for deaf and mute individuals by converting sign language gestures into text and speech using an AI-powered system.

## 🚀 Features
- Real-time sign language detection
- Converts gestures into text and speech
- Uses a CNN-based deep learning model
- Works under different lighting conditions
- User-friendly GUI using Tkinter

## 📂 Project Structure
```
├── final_pred.py          # Main script for sign detection
├── requirements.txt       # List of dependencies
├── app.py                 # GUI-based application
├── test.py                # Script for model testing
├── cnn8grps_rad1_model.h5 # Trained CNN model
├── README.md              # Project documentation
└── Sign_Language_Presentation.pptx  # Project presentation
```

## 🛠 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-link.git
   cd sign-language-conversion
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## 📊 Model Training
- Dataset: Collected and preprocessed images of hand gestures
- Model: Convolutional Neural Network (CNN)
- Accuracy: 97% (tested in various environments)

## 🔍 How It Works
1. Webcam captures hand gestures
2. Preprocessing using OpenCV & MediaPipe
3. CNN model predicts the corresponding letter/word
4. Text and speech output is generated

## 🏆 Future Enhancements
- Support for Indian Sign Language (ISL)
- Mobile app version
- Integration with voice assistants

## 🙌 Acknowledgments
Thanks to the open-source community and researchers working on sign language recognition.

## 📩 Contact
For any queries or collaborations, reach out at: **your-email@example.com**
