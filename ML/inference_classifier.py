# import pickle

# import cv2
# import mediapipe as mp
# import numpy as np

# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {0: 'A', 1: 'B', 2: 'L'}
# while True:

#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10

#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         # Normalize data_aux
#         data_aux = np.array(data_aux)
#         data_aux = (data_aux - np.min(data_aux)) / (np.max(data_aux) - np.min(data_aux) + 1e-6)

#         # Debugging: Print model prediction probabilities
#         prediction = model.predict([data_aux])
#         print(f"Raw prediction output: {prediction}")  # Check the raw output of the model

#         # Ensure prediction is converted to an integer index
#         if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
#             predicted_index = int(np.argmax(prediction[0]))  # Handle multi-dimensional output
#         else:
#             predicted_index = int(np.argmax(prediction))  # Handle flat output

#         predicted_character = labels_dict.get(predicted_index, "Unknown")  # Handle invalid indices

#         print(f"Predicted index: {predicted_index}, Predicted character: {predicted_character}")  # Debugging

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)

#     cv2.imshow('frame', frame)
#     cv2.waitKey(5)


# cap.release()
# cv2.destroyAllWindows()




import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Allow continuous tracking
    min_detection_confidence=0.7,  # Increase confidence threshold for better accuracy
    model_complexity=1
)

DATA_DIR = './data'
IMG_SIZE = 640
NUM_LANDMARKS = 21  # Expected number of hand landmarks

# Ensure dataset directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset directory '{DATA_DIR}' does not exist.")
    exit()

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            continue

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if len(hand_landmarks.landmark) != NUM_LANDMARKS:
                    print(f"Skipping image {img_path} due to inconsistent landmark count.")
                    continue
                
                for i in range(NUM_LANDMARKS):
                    x_.append(hand_landmarks.landmark[i].x * IMG_SIZE)
                    y_.append(hand_landmarks.landmark[i].y * IMG_SIZE)

                min_x, min_y = min(x_), min(y_)
                
                for i in range(NUM_LANDMARKS):
                    data_aux.append(hand_landmarks.landmark[i].x * IMG_SIZE - min_x)
                    data_aux.append(hand_landmarks.landmark[i].y * IMG_SIZE - min_y)

                data.append(data_aux)
                labels.append(int(dir_))  # Ensure labels are stored as integers

if data and labels:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': np.array(data, dtype=np.float32), 'labels': np.array(labels)}, f)
    print("Dataset successfully saved as data.pickle")
    
    # Debugging: Print prediction details
    prediction = model.predict([np.asarray(data_aux)])
    print(f"Raw prediction output: {prediction}")  # Debugging output

    if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
        predicted_index = int(np.argmax(prediction[0]))  
    else:
        predicted_index = int(np.argmax(prediction))  

    print(f"Predicted index: {predicted_index}")

    labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Define label mappings
    
    if predicted_index in labels_dict:
        predicted_character = labels_dict[predicted_index]
    else:
        predicted_character = "Unknown"  # Handle unknown predictions

    print(f"Final Predicted Character: {predicted_character}")
else:
    print("No valid hand landmarks found. Check your images and detection settings.")
