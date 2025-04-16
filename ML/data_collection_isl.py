import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize webcam
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Directory setup - create if not exists
base_dir = "ISL_dataset"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    char_dir = os.path.join(base_dir, char)
    if not os.path.exists(char_dir):
        os.makedirs(char_dir)

# Current letter and count
c_dir = 'A'
count = 0
if os.path.exists(os.path.join(base_dir, c_dir)):
    count = len(os.listdir(os.path.join(base_dir, c_dir)))

offset = 15
step = 1
flag = False
suv = 0

# Create a white background for skeleton drawing
white = np.ones((400, 400), np.uint8) * 255
white_path = "white.jpg"
cv2.imwrite(white_path, white)

# Create UI with Tkinter
root = tk.Tk()
root.title("ISL Data Collection")
root.geometry("1200x700")

# Left side - camera feed
left_frame = tk.Frame(root, width=600, height=600)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Right side - controls
right_frame = tk.Frame(root, width=400, height=600)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Video feed
video_label = tk.Label(left_frame)
video_label.pack(padx=10, pady=10)

# Skeleton view
skeleton_label = tk.Label(left_frame)
skeleton_label.pack(padx=10, pady=10)

# Status indicators
status_frame = tk.Frame(right_frame)
status_frame.pack(fill=tk.X, padx=10, pady=10)

current_letter_label = tk.Label(status_frame, text=f"Current Letter: {c_dir}", font=("Arial", 14))
current_letter_label.pack(side=tk.LEFT, padx=5)

count_label = tk.Label(status_frame, text=f"Images Captured: {count}", font=("Arial", 14))
count_label.pack(side=tk.LEFT, padx=5)

# Instructions
instruction_frame = tk.Frame(right_frame)
instruction_frame.pack(fill=tk.X, padx=10, pady=10)

instructions = tk.Label(
    instruction_frame, 
    text="Instructions:\n\n"
         "1. Press 'n' to move to next letter\n"
         "2. Press 'a' to start/stop automatic capture\n"
         "3. Press 'c' to capture a single frame\n"
         "4. Press 'Esc' to exit\n\n"
         "Make sure your hand is visible in the camera\n"
         "and positioned correctly for the letter.",
    font=("Arial", 12), 
    justify=tk.LEFT
)
instructions.pack(padx=5, pady=5)

# Control buttons
button_frame = tk.Frame(right_frame)
button_frame.pack(fill=tk.X, padx=10, pady=10)

is_capturing = False

def next_letter():
    global c_dir, count, flag
    c_dir = chr(ord(c_dir) + 1)
    if ord(c_dir) == ord('Z') + 1:
        c_dir = 'A'
    flag = False
    count = len(os.listdir(os.path.join(base_dir, c_dir)))
    current_letter_label.config(text=f"Current Letter: {c_dir}")
    count_label.config(text=f"Images Captured: {count}")

def toggle_capture():
    global flag, suv
    flag = not flag
    suv = 0
    capture_btn.config(text="Stop Capturing" if flag else "Start Capturing")

def capture_single():
    global count
    ret, frame = capture.read()
    if ret:
        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = np.array(frame[y - offset:y + h + offset, x - offset:x + w + offset])
            
            # Draw skeleton
            white = cv2.imread(white_path)
            handz = hd2.findHands(image, draw=True, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']
                os_offset = ((400 - w) // 2) - 15
                os1_offset = ((400 - h) // 2) - 15
                
                # Draw lines for the hand skeleton
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                             (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                             (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                             (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                             (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                             (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                
                # Connect finger bases
                cv2.line(white, (pts[5][0] + os_offset, pts[5][1] + os1_offset), 
                         (pts[9][0] + os_offset, pts[9][1] + os1_offset), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0] + os_offset, pts[9][1] + os1_offset), 
                         (pts[13][0] + os_offset, pts[13][1] + os1_offset), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0] + os_offset, pts[13][1] + os1_offset), 
                         (pts[17][0] + os_offset, pts[17][1] + os1_offset), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os_offset, pts[0][1] + os1_offset), 
                         (pts[5][0] + os_offset, pts[5][1] + os1_offset), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os_offset, pts[0][1] + os1_offset), 
                         (pts[17][0] + os_offset, pts[17][1] + os1_offset), (0, 255, 0), 3)
                
                # Draw points for each joint
                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os_offset, pts[i][1] + os1_offset), 2, (0, 0, 255), 1)
                
                # Save the image
                save_path = os.path.join(base_dir, c_dir, f"{count}.jpg")
                cv2.imwrite(save_path, white)
                count += 1
                count_label.config(text=f"Images Captured: {count}")
                
                # Show feedback
                status_label.config(text=f"Captured image {count} for letter {c_dir}")

def update_frame():
    global count, step, suv, flag
    
    try:
        ret, frame = capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            hands = hd.findHands(frame, draw=True, flipType=True)
            
            # Convert OpenCV image to PIL format for Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            
            # Status text overlay on frame
            cv2.putText(frame, f"Letter: {c_dir} Count: {count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Handle hand detection for skeleton
            white_img = None
            
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                image = np.array(frame[y - offset:y + h + offset, x - offset:x + w + offset])
                
                # Draw skeleton
                white = cv2.imread(white_path)
                handz = hd2.findHands(image, draw=False, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']
                    os_offset = ((400 - w) // 2) - 15
                    os1_offset = ((400 - h) // 2) - 15
                    
                    # Draw lines for the hand skeleton
                    for t in range(0, 4, 1):
                        cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                                (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                                (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                                (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                                (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), 
                                (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                    
                    # Connect finger bases
                    cv2.line(white, (pts[5][0] + os_offset, pts[5][1] + os1_offset), 
                            (pts[9][0] + os_offset, pts[9][1] + os1_offset), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0] + os_offset, pts[9][1] + os1_offset), 
                            (pts[13][0] + os_offset, pts[13][1] + os1_offset), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0] + os_offset, pts[13][1] + os1_offset), 
                            (pts[17][0] + os_offset, pts[17][1] + os1_offset), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os_offset, pts[0][1] + os1_offset), 
                            (pts[5][0] + os_offset, pts[5][1] + os1_offset), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os_offset, pts[0][1] + os1_offset), 
                            (pts[17][0] + os_offset, pts[17][1] + os1_offset), (0, 255, 0), 3)
                    
                    # Draw points for each joint
                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os_offset, pts[i][1] + os1_offset), 2, (0, 0, 255), 1)
                    
                    # Convert to PIL for Tkinter
                    white_rgb = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
                    white_img = Image.fromarray(white_rgb)
                    white_imgtk = ImageTk.PhotoImage(image=white_img)
                    skeleton_label.imgtk = white_imgtk
                    skeleton_label.configure(image=white_imgtk)
                    
                    # Auto-capture functionality
                    if flag:
                        if suv >= 180:  # Limit to 180 images per letter
                            flag = False
                            capture_btn.config(text="Start Capturing")
                            status_label.config(text=f"Completed capturing 180 images for letter {c_dir}")
                        
                        if step % 3 == 0:
                            save_path = os.path.join(base_dir, c_dir, f"{count}.jpg")
                            cv2.imwrite(save_path, white)
                            count += 1
                            suv += 1
                            count_label.config(text=f"Images Captured: {count}")
                            status_label.config(text=f"Auto-captured image {count} for letter {c_dir}")
                        
                        step += 1
            
            # Keyboard event handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                root.destroy()
            elif key == ord('n'):
                next_letter()
            elif key == ord('a'):
                toggle_capture()
            elif key == ord('c'):
                capture_single()
    
    except Exception as e:
        print("Error:", traceback.format_exc())
    
    # Schedule the next frame update
    root.after(10, update_frame)

# Keyboard bindings
root.bind('<Escape>', lambda e: root.destroy())
root.bind('n', lambda e: next_letter())
root.bind('a', lambda e: toggle_capture())
root.bind('c', lambda e: capture_single())

# Buttons
next_btn = tk.Button(button_frame, text="Next Letter (n)", command=next_letter, font=("Arial", 12))
next_btn.pack(fill=tk.X, padx=5, pady=5)

capture_btn = tk.Button(button_frame, text="Start Capturing (a)", command=toggle_capture, font=("Arial", 12))
capture_btn.pack(fill=tk.X, padx=5, pady=5)

single_capture_btn = tk.Button(button_frame, text="Single Capture (c)", command=capture_single, font=("Arial", 12))
single_capture_btn.pack(fill=tk.X, padx=5, pady=5)

exit_btn = tk.Button(button_frame, text="Exit (Esc)", command=root.destroy, font=("Arial", 12))
exit_btn.pack(fill=tk.X, padx=5, pady=5)

# Status label at the bottom
status_label = tk.Label(right_frame, text="Ready to capture ISL dataset", font=("Arial", 10))
status_label.pack(padx=10, pady=10)

# Start the video update loop
update_frame()

# Run the main Tkinter loop
root.mainloop()

# Clean up
capture.release()
cv2.destroyAllWindows() 