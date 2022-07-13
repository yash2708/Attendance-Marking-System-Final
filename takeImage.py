import cv2
import os
import csv
import findEyes
def image(cam,detector,path,Name,Enrollment,i):
    sampleNum = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            (eyes_frame,x1,y1,w1,h1)=findEyes.eye(img,x,y,w,h)
            if eyes_frame is None:
                continue
            sampleNum = sampleNum + 1
            cv2.imwrite(
                f"{path}\\"
                + Name
                + "_"
                + Enrollment
                + "_"
                + str(sampleNum+i)
                + ".jpg",
                eyes_frame,
            )
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(img, (x+x1, y+y1), (x+x1 + w1, y+ y1 + h1), (0, 255, 0), 2)
            cv2.imshow("Frame", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif sampleNum > 50:
            break


def TakeImage(cam,haarcasecade_path_face, trainimage_path):
    name = input("Enter Student Name:     ")
    erno = input("Enter Enrolment Number: ")
    if name != "" and erno != "" :
        try:
            cam = cv2.VideoCapture(cam)
            detector = cv2.CascadeClassifier(haarcasecade_path_face)
            Enrollment = erno
            Name = name
            directory = Enrollment + "_" + Name
            path = os.path.join(trainimage_path,directory)
            try:
                os.mkdir(path)
                i=0
            except FileExistsError as F:
                i=len(os.listdir(path))+1
            image(cam,detector,path,Name,Enrollment,i)
            cam.release()
            cv2.destroyAllWindows()
            row = [Enrollment, Name]
            if i==0:
                with open("StudentDetails//studentdetails.csv",
                    "a+",
                ) as csvFile:
                    writer = csv.writer(csvFile, delimiter=",")
                    writer.writerow(row)
                    csvFile.close()
            print("Images Saved for ER No:" + Enrollment + " Name:" + Name)
        except:
            print("Unknown Error")
