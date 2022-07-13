import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import cv2
import numpy as np
def classifier_softmax_regressor(inputDim,labels):
    inputDim=int(inputDim)
    classifier_model=Sequential()
    classifier_model.add(Dense(units=100,input_dim=inputDim,kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.3))
    classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.2))
    classifier_model.add(Dense(units=labels,kernel_initializer='he_uniform'))
    classifier_model.add(Activation('softmax'))
    classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])
    return classifier_model


def classify(frame,vgg_face,classifier_model):
    if frame is not None:
        img=cv2.resize(frame,(224,224))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = img_to_array(img)
        image_array_expanded = np.expand_dims(img_array, axis = 0)
        img=preprocess_input(image_array_expanded)
        img_encode=vgg_face(img)
        embed=K.eval(img_encode)
        person=classifier_model.predict(embed)
        id=np.argmax(person)
        return id,np.max(person)

def classify1(frame,classifier_model):
    if frame is not None:
        #img=frame
        img=cv2.resize(frame,(224,224))
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        recognizer.train([img],np.array([0]))
        img1=recognizer.getHistograms()
        img1=np.array(img1[0])

        person=classifier_model.predict(img1)
        return np.argmax(person),np.max(person)

# img=cv2.imread("D:/ELC_Intern/ELC/ELC_CS/TrainingImage/101903341_ArshJain/ ArshJain_101903341_30.jpg")
# #print(img)
# classifier_model=tf.keras.models.load_model('face_classifier_model_LBPH.h5')
# print(classify1(img,classifier_model))






