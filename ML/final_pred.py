# Importing Libraries
import numpy as np
import math
import cv2

import os, sys
import traceback
import pyttsx3
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase, ascii_lowercase  # Import lowercase letters
import enchant
import time  # Added import statement

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
import tkinter as tk
from PIL import Image, ImageTk

offset=29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# Application :

class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        # Load ASL model
        self.asl_model = load_model(r'C:\Users\batman\Downloads\Sign-Language-To-Text-and-Speech-Conversion-master\cnn8grps_rad1_model.h5')
        # Path for ISL model - this should be replaced with the actual path when available
        self.isl_model_path = r'C:\Users\batman\Downloads\Sign-Language-To-Text-and-Speech-Conversion-master\your_model.h5'
        self.isl_model = load_model(self.isl_model_path) if os.path.exists(self.isl_model_path) else None
        
        # Current active model - default to ASL
        self.current_model = self.asl_model
        # Flag to track which language is currently active
        self.is_asl_active = True
        
        self.speak_engine=pyttsx3.init()
        self.speak_engine.setProperty("rate",100)
        voices=self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice",voices[0].id)

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag=False
        self.next_flag=True
        self.prev_char=""
        self.count=-1
        self.ten_prev_char=[" "] * 10

        # Initialize counters for both ASL and ISL
        for i in ascii_uppercase + ascii_lowercase:
            self.ct[i] = 0
        print("Loaded model from disk")


        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1300x700")
        self.root.configure(bg='black')  # Change background to black

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=3, width=480, height=640)
        self.panel.config(bg='black')

        self.panel2 = tk.Label(self.root)  # initialize image panel
        self.panel2.place(x=700, y=115, width=400, height=400)
        self.panel2.config(bg='black')

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"), bg='black', fg='white')

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=280, y=585)
        self.panel3.config(bg='black')

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=580)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"), bg='black', fg='white')

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=260, y=632)
        self.panel5.config(bg='black')

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=632)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"), bg='black', fg='white')

        self.T4 = tk.Label(self.root)
        self.T4.place(x=10, y=700)
        self.T4.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"), bg='black')


        self.b1=tk.Button(self.root)
        self.b1.place(x=390,y=700)

        self.b2 = tk.Button(self.root)
        self.b2.place(x=590, y=700)

        self.b3 = tk.Button(self.root)
        self.b3.place(x=790, y=700)

        self.b4 = tk.Button(self.root)
        self.b4.place(x=990, y=700)

        self.speak = tk.Button(self.root)
        self.speak.place(x=1305, y=630)
        self.speak.config(text="Speak", font=("Courier", 20), wraplength=100, command=self.speak_fun)

        self.clear = tk.Button(self.root)
        self.clear.place(x=1205, y=630)
        self.clear.config(text="Clear", font=("Courier", 20), wraplength=100, command=self.clear_fun)

        # Toggle button for ASL/ISL
        self.toggle_language = tk.Button(self.root)
        self.toggle_language.place(x=1105, y=630)
        self.toggle_language.config(text="ASL", font=("Courier", 20), wraplength=100, command=self.toggle_language_fun, bg="green")

        # Language indicator label
        self.language_indicator = tk.Label(self.root)
        self.language_indicator.place(x=1105, y=580)
        self.language_indicator.config(text="Current: ASL", font=("Courier", 15), bg='black', fg='green')

        self.str = " "
        self.ccc=0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"


        self.word1=" "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_interval = 1  # seconds

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            cv2image = cv2.flip(frame, 1)
            if cv2image.any:
                hands = hd.findHands(cv2image, draw=False, flipType=True)
                cv2image_copy = np.array(cv2image)
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=self.current_image)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk, bg='black')

                if hands[0]:
                    hand = hands[0]
                    map = hand[0]
                    x, y, w, h = map['bbox']
                    image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                    black = np.zeros((400, 400, 3), dtype=np.uint8)
                    if image.all:
                        handz = hd2.findHands(image, draw=False, flipType=True)
                        self.ccc += 1
                        if handz[0]:
                            hand = handz[0]
                            handmap = hand[0]
                            self.pts = handmap['lmList']

                            os = ((400 - w) // 2) - 15
                            os1 = ((400 - h) // 2) - 15
                            for t in range(0, 4, 1):
                                cv2.line(black, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 128, 0), 3)  # Dark green color
                            for t in range(5, 8, 1):
                                cv2.line(black, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 128, 0), 3)  # Dark green color
                            for t in range(9, 12, 1):
                                cv2.line(black, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 128, 0), 3)  # Dark green color
                            for t in range(13, 16, 1):
                                cv2.line(black, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 128, 0), 3)  # Dark green color
                            for t in range(17, 20, 1):
                                cv2.line(black, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1), (0, 128, 0), 3)  # Dark green color
                            cv2.line(black, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0), 3)  # Light green color
                            cv2.line(black, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0), 3)  # Light green color
                            cv2.line(black, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0), 3)  # Light green color
                            cv2.line(black, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0), 3)  # Light green color
                            cv2.line(black, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0), 3)  # Light green color

                            for i in range(21):
                                cv2.circle(black, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 128, 0), 1)  # Dark green color

                            res = black
                            current_time = time.time()
                            if current_time - self.last_prediction_time > self.prediction_interval:
                                self.predict(res)
                                self.last_prediction_time = current_time

                            self.current_image2 = Image.fromarray(res)

                            imgtk = ImageTk.PhotoImage(image=self.current_image2)

                            self.panel2.imgtk = imgtk
                            self.panel2.config(image=imgtk, bg='black')

                            self.panel3.config(text=self.current_symbol, font=("Courier", 30), fg='cyan')  # Colorful character

                            self.b1.config(text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
                            self.b2.config(text=self.word2, font=("Courier", 20), wraplength=825, command=self.action2)
                            self.b3.config(text=self.word3, font=("Courier", 20), wraplength=825, command=self.action3)
                            self.b4.config(text=self.word4, font=("Courier", 20), wraplength=825, command=self.action4)

                self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025, fg='cyan')  # Colorful sentence
        except Exception as e:
            print(f"Error in video loop: {e}")
            print(traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()


    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str=self.str[:idx_word]
        self.str=self.str+self.word2.upper()
        #self.str[idx_word:last_idx] = self.word2


    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()



    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()


    def speak_fun(self):
        if len(self.str.strip()) > 0:
            if self.is_asl_active:
                self.speak_engine.say("In American Sign Language: " + self.str.strip())
            else:
                self.speak_engine.say("In Indian Sign Language: " + self.str.strip())
            self.speak_engine.runAndWait()


    def clear_fun(self):
        self.str=" "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def toggle_language_fun(self):
        self.is_asl_active = not self.is_asl_active
        
        if self.is_asl_active:
            self.current_model = self.asl_model
            self.toggle_language.config(text="ASL", bg="green")
            self.language_indicator.config(text="Current: ASL", fg="green")
        else:
            # Check if ISL model exists
            if self.isl_model is None:
                if os.path.exists(self.isl_model_path):
                    self.isl_model = load_model(self.isl_model_path)
                    self.current_model = self.isl_model
                    self.toggle_language.config(text="ISL", bg="blue")
                    self.language_indicator.config(text="Current: ISL", fg="blue")
                else:
                    # If ISL model doesn't exist, stay with ASL
                    self.is_asl_active = True
                    self.current_model = self.asl_model
                    self.toggle_language.config(text="ASL", bg="green")
                    self.language_indicator.config(text="Current: ASL (ISL model not found)", fg="red")
            else:
                self.current_model = self.isl_model
                self.toggle_language.config(text="ISL", bg="blue")
                self.language_indicator.config(text="Current: ISL", fg="blue")
                
        # Clear current text when switching languages
        self.clear_fun()

    def predict(self, test_image):
        black = test_image
        black = black.reshape(1, 400, 400, 3)
        prob = np.array(self.current_model.predict(black)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)

        pl = [ch1, ch2]

        # Use different prediction logic based on active language
        if self.is_asl_active:
            # ASL prediction logic
            # condition for [Aemnst]
            l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                 [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                 [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
            if pl in l:
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 0

            # condition for [o][s]
            l = [[2, 2], [2, 1]]
            if pl in l:
                if (self.pts[5][0] < self.pts[4][0]):
                    ch1 = 0

            # condition for [c0][aemnst]
            l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                    ch1 = 2

            # condition for [c0][aemnst]
            l = [[6, 0], [6, 6], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[8], self.pts[16]) < 52:
                    ch1 = 2

            # condition for [gh][bdfikruvw]
            l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                    ch1 = 3

            # con for [gh][l]
            l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][0] > self.pts[0][0]:
                    ch1 = 3

            # con for [gh][pqz]
            l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[2][1] + 15 < self.pts[16][1]:
                    ch1 = 3

            # con for [l][x]
            l = [[6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[4], self.pts[11]) > 55:
                    ch1 = 4

            # con for [l][d]
            l = [[1, 4], [1, 6], [1, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(self.pts[4], self.pts[11]) > 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 4

            # con for [l][gh]
            l = [[3, 6], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[4][0] < self.pts[0][0]):
                    ch1 = 4

            # con for [l][c0]
            l = [[2, 2], [2, 5], [2, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[1][0] < self.pts[12][0]):
                    ch1 = 4

            # con for [l][c0]
            l = [[2, 2], [2, 5], [2, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[1][0] < self.pts[12][0]):
                    ch1 = 4

            # con for [gh][z]
            l = [[3, 6], [3, 5], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                    ch1 = 5

            # con for [gh][pq]
            l = [[3, 2], [3, 1], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                    ch1 = 5

            # con for [l][pqz]
            l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][0] > self.pts[0][0]:
                    ch1 = 5

            # con for [pqz][aemnst]
            l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                    ch1 = 5

            # con for [pqz][yj]
            l = [[5, 7], [5, 2], [5, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[3][0] < self.pts[0][0]:
                    ch1 = 7

            # con for [l][yj]
            l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] < self.pts[8][1]:
                    ch1 = 7

            # con for [x][yj]
            l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[18][1] > self.pts[20][1]:
                    ch1 = 7

            # condition for [x][aemnst]
            l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] > self.pts[16][0]:
                    ch1 = 6

            # condition for [yj][x]
            l = [[7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                    ch1 = 6

            # condition for [c0][x]
            l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[8], self.pts[16]) > 50:
                    ch1 = 6

            # con for [l][x]
            l = [[4, 6], [4, 2], [4, 1], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[4], self.pts[11]) < 60:
                    ch1 = 6

            # con for [x][d]
            l = [[1, 4], [1, 6], [1, 0], [1, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                    ch1 = 6

            # con for [b][pqz]
            l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
                 [6, 3], [6, 4], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            # con for [f][pqz]
            l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
                 [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            # con for [d][pqz]
            fg = 19
            l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                    ch1 = 1

            l = [[4, 1], [4, 2], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(self.pts[4], self.pts[11]) < 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 1

            l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                    ch1 = 1

            l = [[6, 6], [6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                    ch1 = 1

            # con for [i][pqz]
            l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                    ch1 = 1

            # con for [yj][bfdi]
            l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[4][0] < self.pts[5][0] + 15) and (
                (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                    ch1 = 7

            # con for [uvr]
            l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                     self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                    ch1 = 1

            # con for [w]
            fg = 13
            l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                        self.pts[0][0] + fg < self.pts[20][0]) and not (
                        self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                    0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                    ch1 = 1

            # con for [w]

            l = [[5, 0], [5, 5], [0, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                    ch1 = 1

            # -------------------------condn for 8 groups  ends

            # -------------------------condn for subgroups  starts
            #
            if ch1 == 0:
                ch1 = 'S'
                if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                    ch1 = 'A'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                    0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                    ch1 = 'T'
                if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                    ch1 = 'E'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                    ch1 = 'M'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                    ch1 = 'N'

            if ch1 == 2:
                if self.distance(self.pts[12], self.pts[4]) > 42:
                    ch1 = 'C'
                else:
                    ch1 = 'O'

            if ch1 == 3:
                if (self.distance(self.pts[8], self.pts[12])) > 72:
                    ch1 = 'G'
                else:
                    ch1 = 'H'

            if ch1 == 7:
                if self.distance(self.pts[8], self.pts[4]) > 42:
                    ch1 = 'Y'
                else:
                    ch1 = 'J'

            if ch1 == 4:
                ch1 = 'L'

            if ch1 == 6:
                ch1 = 'X'

            if ch1 == 5:
                if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                    if self.pts[8][1] < self.pts[5][1]:
                        ch1 = 'Z'
                    else:
                        ch1 = 'Q'
                else:
                    ch1 = 'P'

            if ch1 == 1:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                    1]):
                    ch1 = 'B'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                    1]):
                    ch1 = 'D'
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                    1]):
                    ch1 = 'F'
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                    1]):
                    ch1 = 'I'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                    1]):
                    ch1 = 'W'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                    1]) and self.pts[4][1] < self.pts[9][1]:
                    ch1 = 'K'
                if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                        self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                        self.pts[20][1]):
                    ch1 = 'U'
                if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                        self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                        self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                    ch1 = 'V'

                if (self.pts[8][0] > self.pts[12][0]) and (
                        self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                        self.pts[20][1]):
                    ch1 = 'R'

            if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1=" "



            print(self.pts[4][0] < self.pts[5][0])
            if ch1 == 'E' or ch1=='Y' or ch1=='B':
                if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1="next"


            if ch1 == 'Next' or ch1 == 'B' or ch1 == 'C' or ch1 == 'H' or ch1 == 'F' or ch1 == 'X':
                if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                    ch1 = 'Backspace'


            if ch1=="next" and self.prev_char!="next":
                if self.ten_prev_char[(self.count-2)%10]!="next":
                    if self.ten_prev_char[(self.count-2)%10]=="Backspace":
                        self.str=self.str[0:-1]
                    else:
                        if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                            self.str = self.str + self.ten_prev_char[(self.count-2)%10]
                else:
                    if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]


            if ch1=="  " and self.prev_char!="  ":
                self.str = self.str + "  "

            self.prev_char=ch1
            self.current_symbol=ch1
            self.count += 1
            self.ten_prev_char[self.count%10]=ch1


            if len(self.str.strip())!=0:
                st=self.str.rfind(" ")
                ed=len(self.str)
                word=self.str[st+1:ed]
                self.word=word
                if len(word.strip())!=0:
                    ddd.check(word)
                    lenn = len(ddd.suggest(word))
                    if lenn >= 4:
                        self.word4 = ddd.suggest(word)[3]

                    if lenn >= 3:
                        self.word3 = ddd.suggest(word)[2]

                    if lenn >= 2:
                        self.word2 = ddd.suggest(word)[1]

                    if lenn >= 1:
                        self.word1 = ddd.suggest(word)[0]
                else:
                    self.word1 = " "
                    self.word2 = " "
                    self.word3 = " "
                    self.word4 = " "

        else:
            # ISL prediction logic - this should be customized based on ISL dataset
            # For now, we use the same logic as ASL but would need to be updated with specific ISL hand detection rules
            # This is a placeholder until a proper ISL model and detection logic is implemented
            l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                 [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                 [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
            if pl in l:
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 0

        # Final character output
        if self.is_asl_active:
            # ASL output conversion
            if ch1 == 0:
                ch1 = 'S'
            if ch1 == 2:
                ch1 = 'C'
            if ch1 == 3:
                ch1 = 'G'
            if ch1 == 7:
                ch1 = 'Y'
            if ch1 == 4:
                ch1 = 'L'
            if ch1 == 6:
                ch1 = 'X'
            if ch1 == 5:
                ch1 = 'P'
            if ch1 == 1:
                ch1 = 'B'

        else:
            # ISL output conversion
            # This would need to be customized for ISL alphabet/gestures
            # For now, we're using the same logic as ASL but should be updated
            if ch1 == 0:
                ch1 = 'S'
            if ch1 == 2:
                ch1 = 'C'
            if ch1 == 3:
                ch1 = 'G'
            if ch1 == 7:
                ch1 = 'Y'
            if ch1 == 4:
                ch1 = 'L'
            if ch1 == 6:
                ch1 = 'X'
            if ch1 == 5:
                ch1 = 'P'
            if ch1 == 1:
                ch1 = 'B'

        self.prev_char=ch1
        self.current_symbol=ch1
        self.count += 1
        self.ten_prev_char[self.count%10]=ch1

        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            if len(word.strip())!=0:
                ddd.check(word)
                lenn = len(ddd.suggest(word))
                if lenn >= 4:
                    self.word4 = ddd.suggest(word)[3]

                if lenn >= 3:
                    self.word3 = ddd.suggest(word)[2]

                if lenn >= 2:
                    self.word2 = ddd.suggest(word)[1]

                if lenn >= 1:
                    self.word1 = ddd.suggest(word)[0]
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "


    def destructor(self):
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


labels_dict = {
    0: "HELLO", 1: "I LOVE YOU", 2: "THANK YOU", 3: "PLEASE", 4: "YES", 5: "NO",
    6: "GOOD MORNING", 7: "GOOD NIGHT", 8: "HOW ARE YOU", 9: "BEST OF LUCK",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
    20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T",
    30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z",
    36: "a", 37: "b", 38: "c", 39: "d", 40: "e", 41: "f", 42: "g", 43: "h", 44: "i", 45: "j",
    46: "k", 47: "l", 48: "m", 49: "n", 50: "o", 51: "p", 52: "q", 53: "r", 54: "s", 55: "t",
    56: "u", 57: "v", 58: "w", 59: "x", 60: "y", 61: "z",
    62: "Abbreviation", 63: "Abeyance", 64: "About", 65: "Above", 66: "Absent", 67: "Absorb",
    68: "Absorption", 69: "Abstract Of Account", 70: "Acceleration", 71: "Accept", 72: "Acceptance",
    73: "Acceptance Clean", 74: "Acceptor For Honour", 75: "Access", 76: "Accessories", 77: "Accident",
    78: "Accommodation", 79: "Account", 80: "Account Blocked", 81: "Account Book", 82: "Account Branch Adjustment",
    83: "Account Charges", 84: "Account Closed", 85: "Account Closing", 86: "Account Commission", 87: "Account Currency",
    88: "Account Current", 89: "Account Day", 90: "Account Discount", 91: "Account Govemment", 92: "Account Head",
    93: "Account Holder", 94: "Account Inoperative", 95: "Account Joint", 96: "Account Margin", 97: "Account Minor",
    98: "Account Payee Only", 99: "Account Profit And Loss", 100: "Account Rupee", 101: "Account Suit Filed", 102: "Account Suspence",
    103: "Accumulated", 104: "Accurate", 105: "Accured", 106: "Accuse", 107: "Acknowledgement", 108: "Across", 109: "Act", 110: "Acting",
    111: "Action", 112: "Active", 113: "Actor", 114: "Actress", 115: "Actual", 116: "Add", 117: "Additional", 118: "Address-1", 119: "Address-2",
    120: "Adhesive", 121: "Adjustable Screw", 122: "Adjustement", 123: "Admissible", 124: "Advance", 125: "Advantage", 126: "Advertisement-1",
    127: "Advertisement-2", 128: "Advise", 129: "Aeroplane", 130: "Affidavit", 131: "Africa", 132: "After", 133: "Afternoon", 134: "Again",
    135: "Age", 136: "Agenda", 137: "Agra", 138: "Agree", 139: "Agreement", 140: "Agriculture", 141: "Ahemedabad", 142: "Air", 143: "Airmail",
    144: "Airport", 145: "Air Pollution", 146: "Alarm System", 147: "Alive", 148: "All", 149: "Allah", 150: "Allergy", 151: "Allotment",
    152: "Allow", 153: "Allowance", 154: "Alone", 155: "Alternate Current", 156: "Altitude", 157: "Aluminium", 158: "Always", 159: "Amalgamation",
    160: "Ambulance", 161: "Amendments", 162: "Ammeter", 163: "Among", 164: "Amount", 165: "Ampere", 166: "Amplitude", 167: "Amritsar",
    168: "Andhra Pradesh", 169: "Angel", 170: "Angle", 171: "Angle Of Refraction", 172: "Angular Velocity", 173: "Animals", 174: "Anklets",
    175: "Announce", 176: "Annual", 177: "Annual Closing", 178: "Annual Report", 179: "Antartica", 180: "Anticlockwise-1", 181: "Anticlockwise-2",
    182: "Ant-1", 183: "Ant-2", 184: "Any", 185: "Apparatus", 186: "Appear", 187: "Apple", 188: "Applicant",
    189: "Application", 190: "Appointment", 191: "Appreciation", 192: "Approval", 193: "April", 194: "Area", 195: "Argue", 196: "Arm", 197: "Arms And Amunitions",
    198: "Around", 199: "Arrears", 200: "Arrest", 201: "Art", 202: "Artificial", 203: "Asia", 204: "Ask-1", 205: "Ask-2", 206: "Assam", 207: "Assembly-1",
    208: "Assembly-2", 209: "Assets", 210: "Assistance", 211: "Association", 212: "Astronomy", 213: "At", 214: "Atmosphere", 215: "Atom", 216: "Attachmant Of Property",
    217: "Attendant", 218: "Attorney", 219: "Attraction", 220: "Attractive Force", 221: "Audiologist", 222: "Audio Frequency", 223: "Audit", 224: "Auditorium",
    225: "August-1", 226: "August-2", 227: "Australia-1", 228: "Australia-2", 229: "Authorisation", 230: "Authorised Signature", 231: "Automatic", 232: "Automobiles",
    233: "Autorickshaw", 234: "Average", 235: "Avoid", 236: "Axe", 237: "Baby", 238: "Back", 239: "Bad Conductor", 240: "Bad Debts", 241: "Bail", 242: "Bake", 243: "Balance", 244: "Balance Closing",
    245: "Balance Opening", 246: "Balance Outstanding", 247: "Balance Sheet", 248: "Balcony", 249: "Ball", 250: "Balloon", 251: "Banana", 252: "Banaras", 253: "Bandage",
    254: "Bandh", 255: "Bangalore", 256: "Bangladesh", 257: "Bangles", 258: "Banian", 259: "Bank", 260: "Bank Nationalised", 261: "Barber", 262: "Bark", 263: "Bar Magnet",
    264: "Basketball", 265: "Basket Of Currencies", 266: "Bat", 267: "Bathroom", 268: "Bath-1", 269: "Bath-2", 270: "Batminton", 271: "Battery", 272: "Beak", 273: "Beaker",
    274: "Beans", 275: "Bear", 276: "Bearer", 277: "Beat", 278: "Beautiful", 279: "Become", 280: "Bed", 281: "Bedroom", 282: "Bedsheet", 283: "Before", 284: "Begin",
    285: "Behind", 286: "Belcony", 287: "Belief", 288: "Bell-1", 289: "Bell-2", 290: "Below", 291: "Belt", 292: "Bench", 293: "Bend", 294: "Benefit", 295: "Bengali",
    296: "Berth", 297: "Between", 298: "Beverages", 299: "Bhangada", 300: "Bhopal", 301: "Bible", 302: "Biconcave Lens", 303: "Biconvex Lens", 304: "Big", 305: "Bihar",
    306: "Bill", 307: "Bills Purchased", 308: "Bills Receivable", 309: "Bill Of Exchange", 310: "Bill – Clean", 311: "Bill – Dishonoured", 312: "Bill – Inward", 313: "Bill – Outward",
    314: "Bill – Overdue", 315: "Bill – Suit Filed", 316: "Bimetallic", 317: "Binary", 318: "Biomass", 319: "Bio Energy", 320: "Bird", 321: "Birds, Winds, Fly", 322: "Birth",
    323: "Birthday", 324: "Biscuit", 325: "Bite", 326: "Bitter", 327: "Black", 328: "Black Board", 329: "Blind", 330: "Blood", 331: "Blood Pressure", 332: "Blouse", 333: "Blow",
    334: "Blue", 335: "Boat", 336: "Body", 337: "Bogie", 338: "Boil", 339: "Boiling Point", 340: "Bold", 341: "Bolt", 342: "Bond", 343: "Bondage of Nature", 344: "Bone",
    345: "Book", 346: "Boring", 347: "Borrow", 348: "Borrower", 349: "Both Friend and Foe the Saints Adore", 350: "Bottle", 351: "Bowl", 352: "Boxing", 353: "Boy", 354: "Brain",
    355: "Branch", 356: "Brass", 357: "Brave", 358: "Bread Board", 359: "Bread (Pev)-2", 360: "Bread-1", 361: "Break", 362: "Breakfast", 363: "Breathe", 364: "Brick", 365: "Bridge",
    366: "Bright", 367: "Bring", 368: "Brinjal", 369: "Brittle", 370: "Brokerage", 371: "Brother", 372: "Brother-In-Law", 373: "Brown", 374: "Brush", 375: "Bucket", 376: "Bud",
    377: "Buddah", 378: "Buddha Purnima", 379: "Budget", 380: "Buffalo", 381: "Bug", 382: "Build", 383: "Building", 384: "Bulb", 385: "Bullock Cart", 386: "Burglar", 387: "Bury",
    388: "Bus", 389: "Business", 390: "Busy", 391: "Bus Conductor", 392: "Bus Stand", 393: "Butter", 394: "Butterfly", 395: "Buy", 396: "Buyer", 397: "Bye Laws", 398: "Cabbage",
    399: "Cake", 400: "Calculation", 401: "Calculator", 402: "Calcutta", 403: "Call", 404: "Call Deposit", 405: "Call Money", 406: "Calm", 407: "Calorie", 408: "Calorimeter",
    409: "Camel", 410: "Camera", 411: "Can", 412: "Cancel", 413: "Cancer", 414: "Candle", 415: "Cap", 416: "Capacitor-1", 417: "Capacitor-2", 418: "Capacity", 419: "Capillary Tube",
    420: "Capital", 421: "Car", 422: "Carburetor", 423: "Card", 424: "Careful-1", 425: "Careful-2", 426: "Careless", 427: "Carpenter", 428: "Carrom", 429: "Carrot", 430: "Carry",
    431: "Cash", 432: "Cash Book", 433: "Cash Certificates", 434: "Cash Credit", 435: "Cash Credit Key", 436: "Cash Receipt", 437: "Cash Remmitance", 438: "Cassette", 439: "Catch",
    440: "Cat-1", 441: "Cat-2", 442: "Cauliflower", 443: "Cave", 444: "Celebrate", 445: "Celsius", 446: "Cement", 447: "Center", 448: "Centigram", 449: "Centimeter", 450: "Centimetre",
    451: "Centre Of Curvature", 452: "Centrifugal", 453: "Centripetal", 454: "Certificate-1", 455: "Certificate-2", 456: "Chain Reaction", 457: "Chair", 458: "Chalk", 459: "Chandigarh",
    460: "Change", 461: "Change In State", 462: "Change In Temperature", 463: "Chapathi", 464: "Chapter", 465: "Charge", 466: "Charge Sheet", 467: "Chart", 468: "Chase", 469: "Chat",
    470: "Cheap", 471: "Cheat", 472: "Chemical Energy", 473: "Chemistry", 474: "Chennai-1", 475: "Chennai-2", 476: "Cheque Crossed", 477: "Cheque Endorsement", 478: "Cheque Fraudulent",
    479: "Cheque Gift", 480: "Cheque Mutilated", 481: "Cheque Postdated", 482: "Cheque Returned Memo", 483: "Cheque Stale", 484: "Cheque Travellers", 485: "Cheque-1", 486: "Cheque-2",
    487: "Chess", 488: "Chest", 489: "Chew", 490: "Chicken", 491: "Chickoo", 492: "Chief", 493: "Chief Minister", 494: "Children", 495: "Chillie", 496: "China-1", 497: "China-2",
    498: "Chocolate", 499: "Cholera-1", 500: "Cholera-2", 501: "Christian-1", 502: "Christian-2", 503: "Christmas-1", 504: "Christmas-2", 505: "Church", 506: "Cinema", 507: "Circle",
    508: "Circuit", 509: "Circular", 510: "Circular Motion", 511: "Circus", 512: "City", 513: "Claim", 514: "Clap", 515: "Classification", 516: "Class Room", 517: "Clean", 518: "Clearing House",
    519: "Clerk-1", 520: "Clerk-2", 521: "Clever", 522: "Client", 523: "Climb Down", 524: "Climb Up", 525: "Clinic", 526: "Clinical Thermometer", 527: "Clock", 528: "Clockwise", 529: "Close",
    530: "Cloth", 531: "Clouds", 532: "Clown", 533: "Coal", 534: "Coat", 535: "Cobbler", 536: "Cochin", 537: "Cock", 538: "Cockroach", 539: "Coconut-1", 540: "Coconut-2", 541: "Coded Message",
    542: "Coffee", 543: "Coherent", 544: "Cohesive", 545: "Coimbatore", 546: "Coin", 547: "Cold", 548: "Cold Drink", 549: "Collecting Agent", 550: "College", 551: "Collision", 552: "Colour Filter",
    553: "Comb", 554: "Come", 555: "Comfortable", 556: "Committee", 557: "Common", 558: "Communicate", 559: "Communication", 560: "Compact Disc", 561: "Company", 562: "Company Joint Stock"
}
print("Starting Application...")

(Application()).root.mainloop()