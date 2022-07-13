import cv2
import pandas as pd
import time
import datetime
import findEyes
import os
import tensorflow as tf
import classifier_model as classifymodel
def attendance(haarcasecade_path,studentdetail_path,attendance_path):
    sub = input("Enter Subject Name:   ")
    facecasCade = cv2.CascadeClassifier(haarcasecade_path)
    df = pd.read_csv(studentdetail_path)
    cam = cv2.VideoCapture(0)
    now = time.time()
    future = now + 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ["Enrollment", "Name"]
    attendance = pd.DataFrame(columns=col_names)
    classifier_model=tf.keras.models.load_model('face_classifier_model_LBPH.h5')
    while True:
        ___, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = facecasCade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            global Id
            eyes_frame,x1,y1,w1,h1=findEyes.eye(im,x,y,w,h)
            if eyes_frame is None:
                continue
            Id,conf=classifymodel.classify1(eyes_frame,classifier_model)
            print("id:",Id , conf)
            if conf > 0.50: 
                global Subject
                global aa
                global date
                global timeStamp
                Subject = sub
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                aa = df.loc[df["Enrollment"] == Id]["Name"].values
                global tt
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id,aa,]
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 4)
                cv2.rectangle(im, (x+ x1,y+  y1), (x + x1 + w1,y + y1 + h1), (0, 260, 0), 4)
                cv2.putText(im, str(tt), (x1 + h1, y1), font, 1, (255, 255, 0,), 4)
            else:
                Id = "Unknown"
                tt = str(Id)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
                cv2.putText(im, str(tt), (x + h, y), font, 1, (0, 25, 255), 4)
            if time.time() > future:
                break

        attendance = attendance.drop_duplicates(["Enrollment"], keep="first")
        cv2.imshow("Filling Attendance...", im)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        ts = time.time()
    attendance[date] = 1
    date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    Hour, Minute, Second = timeStamp.split(":")
    path = os.path.join(attendance_path, Subject)
    fileName = (f"{path}/"+ Subject+ "_"+ date+ "_"+ Hour+ "-"+ Minute+ "-"+ Second+ ".csv")
    attendance = attendance.drop_duplicates(["Enrollment"], keep="first")
    print(attendance)
    attendance.to_csv(fileName, index=False)
    print("Attendance Filled Successfully of " + Subject)
    cam.release()
    cv2.destroyAllWindows()