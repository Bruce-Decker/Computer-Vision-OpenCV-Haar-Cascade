import cv2
import numpy as np
from PIL import ImageGrab
import sys
import time


def process_img(image):
    original_image = image
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1= 200, threshold2 = 300)
    height, width = processed_img.shape[:2] 
    vertices = np.array([[0, 720],[0, 500], [500, 400], [780, 400], [1280, 500], [1280, 720],], np.int32)
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    processed_img = roi(processed_img, [vertices])
    
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 20, 15)
    return processed_img

def draw_lines(img, lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
        
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


human_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')
cap = cv2.VideoCapture('Driving.m4v')
while cap.isOpened():
    

    ret, frame1 = cap.read()
    ret, frame_original = cap.read()
    frame_oringal = cv2.resize(frame_original, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    frame1 = cv2.resize(frame1, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
   
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    humans = human_classifier.detectMultiScale(gray1, 1.2, 5)
    cars = car_classifier.detectMultiScale(gray1, 1.2, 1)
    
  
    for (x,y,w,h) in humans:
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 255), 2)
     
    for (x,y,w,h) in cars:
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 3)
    screen = frame_original
    new_screen = process_img(screen)
    cv2.imshow('window', new_screen)
    cv2.imshow('Cars', frame1)
    
    if cv2.waitKey(1) == 13 & 0xFF == ord('q'): #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
