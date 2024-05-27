import cv2    
import time
import pandas as pd
from ultralytics import YOLO
import os
import cvzone
import numpy as np
import pytesseract
from vidgear.gears import CamGear
from tracker import*
from math import dist
from datetime import datetime
cpt = 0
maxFrames = 100 # if you want 5 frames only.

count=0
cap=cv2.VideoCapture('khid.mp4')
while cpt < maxFrames:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1080,500))
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"F:\Desktop\carnumberplate-main\images10\numberplate_%d.jpg" %cpt, frame)
    time.sleep(0.01)
    cpt += 1
###################################################################################
image_directory ="images1"  # Replace with the actual path to your image directory

# Get a list of image files
image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Iterate through each image file
for image_file in image_files:
    image_name = os.path.splitext(image_file)[0]  # Extract the image name without extension
    txt_file = image_name + ".txt"  # Assume the corresponding text file has the same name with .txt extension

    # Check if the corresponding text file exists
    if not os.path.exists(os.path.join(image_directory, txt_file)):
        # Delete the image file if the corresponding text file does not exist
        os.remove(os.path.join(image_directory, image_file))
        print(f"Deleted {image_file} because {txt_file} does not exist.")
    

##############################################################################
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('best4.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('khid.mp4')

my_file = open("coco2.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

area = [(35, 375), (16, 456), (1015, 451), (965, 378)]

count = 0
list1 = []
processed_numbers = set()

# Open file for writing car plate data
with open("car_plate_data.txt", "a") as file:
    file.write("NumberPlate\tDate\tTime\n")  # Writing column headers

while True:    
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
   
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
   
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if result >= 0:
           crop = frame[y1:y2, x1:x2]
           gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
           gray = cv2.bilateralFilter(gray, 10, 20, 20)

           text = pytesseract.image_to_string(gray).strip()
           text = text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
           if text not in processed_numbers:
              processed_numbers.add(text) 
              list1.append(text)
              current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
              with open("car_plate_data.txt", "a") as file:
                   file.write(f"{text}\t{current_datetime}\n")
                   cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                   cv2.imshow('crop', crop)

      
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()    
cv2.destroyAllWindows()
