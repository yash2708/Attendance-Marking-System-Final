import cv2
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml") 
def eye(img,x,y,w,h):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        ex=eyes[0][0]-15
        ey=eyes[0][1]-8
        ew=eyes[1][0]-eyes[0][0]+eyes[1][2]+30
        eh=max(eyes[0][3],eyes[1][3])  + 3
        if ew*1.0/eh < 2:
            return None,None,None,None,None
        eyes_frame=roi_color[ey:ey+eh,ex:ex+ew]
        eyes_frame=cv2.resize(eyes_frame,(224,224))
        return eyes_frame,ex,ey,ew,eh
    except:
        return None,None,None,None,None


