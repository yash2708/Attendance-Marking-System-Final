import csv
import os, cv2
import numpy as np
import pandas as pd
import datetime
import time
from PIL import ImageTk, Image
import URLS
import classifier_model as classifyModel
from sklearn import preprocessing
import tensorflow as tf
def TrainImage_LBHP_Softmax(trainimage_path):
    label_encoder = preprocessing.LabelEncoder()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Id = getImagesAndLables(trainimage_path)
    recognizer.train(faces, np.array(Id))
    mat=recognizer.getHistograms()
    x_train=[]
    for i in mat:
        ls=[]
        for j in i[0]:
            ls.append(j)
        x_train.append(ls)
    x_train=np.array(x_train)
    y_train=label_encoder.fit_transform(np.array(Id))
    labels={}
    for i in range(len(Id)):
        labels[y_train[i]]=Id[i]
    print(labels)
    model = classifyModel.classifier_softmax_regressor(16384,len(labels))
    model.fit(x_train,y_train,epochs=30)
    tf.keras.models.save_model(model,'face_classifier_model_LBPH.h5')
    res = "Image Trained successfully"
    print(res)

def getImagesAndLables(path):
    newdir = [os.path.join(path, d) for d in os.listdir(path)]
    imagePath = [
        os.path.join(newdir[i], f)
        for i in range(len(newdir))
        for f in os.listdir(newdir[i])
    ]
    faces = []
    Ids = []
    for imagePath in imagePath:
        pilImage = Image.open(imagePath).convert("L")
        #pilImage = cv2.cvtColor(pilImage, cv2.COLOR_BGR2GRAY)
        imageNp = np.array(pilImage, "uint8")
        #imageNp = cv2.cvtColor(imageNp, cv2.COLOR_BGR2GRAY)
        Id = int(os.path.split(imagePath)[-1].split("_")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids
