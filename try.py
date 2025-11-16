import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector


model = load_model(r"C:\Users\Asus\Desktop\language\levi.h5")


with open("Model/labels.txt", "r") as f:
    labels = f.read().splitlines()


detector = HandDetector(maxHands=1)


cap = cv2.VideoCapture(0)
imgSize = 224

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        
        imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]

        
        imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))

        
        imgArray = np.array(imgCrop) / 255.0

        
        imgArray = np.expand_dims(imgArray, axis=0)

        
        prediction = model.predict(imgArray)

    
        index = np.argmax(prediction)  
        label = labels[index]

        
        cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow("Sign Language Detection", img)

    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()  
cv2.destroyAllWindows()
