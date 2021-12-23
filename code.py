import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

X,y = fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

xtrain, xtest, ytrain,ytest = train_test_split(X,y, random_state= 9,train_size=7500,test_size=2500)
xtrainscale=xtrain/255.0
xtestscale = xtest/255.0
model = LogisticRegression(solver='saga', multi_class='multinomial').fit(xtrainscale,ytrain)
ypred = model.predict(xtestscale)
accuracy = accuracy_score(ytest,ypred)
print(accuracy)

cap=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upperleft = (int(width/2-56),int(height/2-56))
        bottomright = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi = gray[upperleft[1]:bottomright[1], upperleft[0]:bottomright[0]]
        imagepil = Image.fromarray(roi)
        imagebw = imagepil.convert('L')
        imagebwresize=imagebw.resize((28,28),Image.ANTIALIAS)
        imagebwresizeinverted=PIL.ImageOps.invert(imagebwresize)
        pixelfilter = 20
        minpixel = np.percentile(imagebwresizeinverted,pixelfilter)
        imagebwresizeinvertedscale=np.clip(imagebwresizeinverted-minpixel,0,255)
        maxpixel = np.max(imagebwresizeinverted)
        imagebwresizeinvertedscale=np.asarray(imagebwresizeinvertedscale)/maxpixel
        testsample = np.array(imagebwresizeinvertedscale).reshape(1,784)
        testpred = model.predict(testsample)
        print("predicted class is: ", testpred)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()
