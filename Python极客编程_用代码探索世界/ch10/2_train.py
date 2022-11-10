import os
import numpy as np
import cv2 as cv

cascade_path = "C:\\ProgramData\\Anaconda3\\envs\\test1\\Lib\\site-packages\\cv2\\data\\"
face_detector = cv.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')

train_path = './demming_trainer'
# train_path = './trainer'
image_paths = [os.path.join(train_path, f) for f in os.listdir(train_path)]
images, labels = [], []

for image in image_paths:
    train_image = cv.imread(image, cv.IMREAD_GRAYSCALE)
    label = int(os.path.split(image)[-1].split('.')[1])
    name = os.path.split(image)[-1].split('.')[0]
    frame_num = os.path.split(image)[1].split('.')[2]
    faces = face_detector.detectMultiScale(train_image)
    for (x, y, w, h) in faces:
        images.append(train_image[y:y+h, x:x+w])
        labels.append(label)
        print(f"Preparing training images for {name}.{label}.{frame_num}")
        cv.imshow("Training Image", train_image[y:y+h. x:x+w])
        cv.waitKey(50)

cv.destroyAllWIndows()

recongnizer = cv.face.LBPHFaceRecognizer_create()
recongnizer.train(images, np.array(labels))
recongnizer.write('lbph_train.yml')
print("Training complete. Exiting...")