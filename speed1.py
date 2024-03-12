import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import*
import time

model=YOLO('best2.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
            
        



cap=cv2.VideoCapture('id4.mp4')


my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
print(class_list)

cy1=394
offset=6
cy10=455

count=0
tracker=Tracker()
tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()
bus={}
busc=[]
car={}
carc=[]
auto_rikshaw={}
auto_rikshawc=[]
motorcycle={}
motorcyclec=[]
while True:    
    ret,frame = cap.read()
    if not ret:
       break
   

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
   

#    print(px)
    list=[]
    list1=[]
    list2=[]
    list3=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        
        d=int(row[5])
        c=class_list[d]
       
        if 'bus' in c:
            list.append([x1,y1,x2,y2])
        elif 'car' in c:
             list1.append([x1,y1,x2,y2])
        elif 'auto-rikshaw' in c:
             list2.append([x1,y1,x2,y2])
        elif 'motor-cycle' in c:
             list3.append([x1,y1,x2,y2])
        
    bbox_idx=tracker.update(list)
    bbox1_idx=tracker1.update(list1)
    bbox2_idx=tracker2.update(list2)
    bbox3_idx=tracker3.update(list3)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if cy1<(cy+offset) and cy1>(cy-offset):
           bus[id]=time.time()
        if id in bus:
           if cy10<(cy+offset) and cy10>(cy-offset):
              elapsed_time=time.time() - bus[id]
              cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
              cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
              if busc.count(id)==0:
                 busc.append(id)
                 distance = 10 # meters
                 a_speed_ms = distance / elapsed_time
                 a_speed_kh = a_speed_ms * 3.6
                 cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)  
############################################################CAR#########################
    for bbox1 in bbox1_idx:
        x5,y5,x6,y6,id1=bbox1
        cx2=int(x5+x6)//2
        cy2=int(y5+y6)//2
        if cy1<(cy2+offset) and cy1>(cy2-offset):
           car[id1]=time.time()
        if id1 in car:
           if cy10<(cy2+offset) and cy10>(cy2-offset):
              elapsed_time=time.time() - car[id1]
              cv2.rectangle(frame,(x5,y5),(x6,y6),(0,255,0),2)
              cvzone.putTextRect(frame,f'{id1}',(x5,y5),1,1)
              if carc.count(id1)==0:
                 carc.append(id1)
                 distance = 10 # meters
                 a_speed_ms = distance / elapsed_time
                 a_speed_kh = a_speed_ms * 3.6
                 cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x6,y6),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)             
###############################################AUTO_RIKSHAW#########################
    for bbox2 in bbox2_idx:
        x7,y7,x8,y8,id2=bbox2
        cx3=int(x7+x8)//2
        cy3=int(y7+y8)//2
        if cy1<(cy3+offset) and cy1>(cy3-offset):
           auto_rikshaw[id2]=time.time()
        if id2 in auto_rikshaw:
           if cy10<(cy3+offset) and cy10>(cy3-offset):
              elapsed_time=time.time() - auto_rikshaw[id2]
              cv2.rectangle(frame,(x7,y7),(x8,y8),(0,255,0),2)
              cvzone.putTextRect(frame,f'{id2}',(x7,y7),1,1)
              if auto_rikshawc.count(id2)==0:
                 auto_rikshawc.append(id2)
                 distance = 10 # meters
                 a_speed_ms = distance / elapsed_time
                 a_speed_kh = a_speed_ms * 3.6
                 cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x8,y8 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)
##############################################MOTOR-CYCLE###########################
    for bbox3 in bbox3_idx:
        x9,y9,x10,y10,id3=bbox3
        cx4=int(x9+x10)//2
        cy4=int(y9+y10)//2
        if cy1<(cy4+offset) and cy1>(cy4-offset):
           motorcycle[id3]=time.time()
        if id3 in motorcycle:
           if cy10<(cy4+offset) and cy10>(cy4-offset):
              elapsed_time=time.time() - motorcycle[id3]
              cv2.rectangle(frame,(x9,y9),(x10,y10),(0,255,0),2)
              cvzone.putTextRect(frame,f'{id3}',(x9,y9),1,1)
              if motorcyclec.count(id3)==0:
                 motorcyclec.append(id3)
                 distance = 10 # meters
                 a_speed_ms = distance / elapsed_time
                 a_speed_kh = a_speed_ms * 3.6
                 cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x10,y10 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)
    countbus=(len(busc))
    countcar=(len(carc))
    countauto_rikshaw=(len(auto_rikshawc))
    countmotorcycle=(len(motorcyclec))
    cvzone.putTextRect(frame,f'buscount:-{countbus}',(50,60),2,2)
    cvzone.putTextRect(frame,f'countcar:-{countcar}',(50,140),2,2)
    cvzone.putTextRect(frame,f'countauto_rikshaw:-{countauto_rikshaw}',(600,60),2,2)
    cvzone.putTextRect(frame,f'countmotorcycle:-{countmotorcycle}',(600,140),2,2)
            
    cv2.line(frame,(388,394),(562,394),(255,0,255),2)
    cv2.line(frame,(330,455),(580,455),(255,255,255),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()    
cv2.destroyAllWindows()




