import os
import pyttsx3
import cv2 as cv
from playsound import playsound

engine = pyttsx3.init()
engine.setProperty('rate', 145)
engine.setProperty('volume', 1.0)

root_dir = os.path.abspath('.')
tone_path = os.path.join(root_dir, 'tone.wav')

path = "C:\\ProgramData\\Anaconda3\\envs\\test1\\Lib\\site-packages\\cv2\\data\\"
face_detector = cv.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Could not open video device.")
cap.set(3, 640)
cap.set(4, 480)

engine.say("Enter your information when prompted on screen. \n" +
           "Then remove glasses and look directly at webcam. \n" +
           "Make multiplu faces including normal, happy, sad, sleepy. \n" +
           "Continue until you hear the tone.")
engine.runAndWait()

name = input("\nEnter last name: ")
user_id = input("Enter assigned ID Number: ")
print("\nCapturing face. Look at the camera now!")

if not os.path.isdir('trainer'):
    os.mkdir('trainer')
os.chdir('trainer')

frame_count = 0
while True:
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rects = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in face_rects:
        frame_count += 1
        cv.imwrite(str(name) + '.' + str(user_id) + '.' + str(frame_count) + '.jpg', gray[y:y+h, x:x+w])
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow('image', frame)
        cv.waitKey(400)
    if frame_count >= 30:
        break

print("\nImage collection complete. Exiting...")
playsound(tone_path, block=False)
cap.release()
cv.destroyAllWindows()