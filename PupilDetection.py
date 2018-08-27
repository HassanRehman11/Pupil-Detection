import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
eye_cascade = cv2.CascadeClassifier('haarcascae_eye.xml')
cap = cv2.VideoCapture('Video.mp4')

diameter=[]
blink =False
bcount = -1
kernel = np.ones((5,5),np.uint8)
global a
font = cv2.FONT_HERSHEY_SIMPLEX
try:
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray,1.1,7)
        if (len(eyes)>0):
            a = "Eye Open"
            
            if (blink==True):
                blink=False
               
            cv2.putText(img,a,(10,30), font, 1,(0,0,255),2,cv2.LINE_AA)
            
            for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                roi_gray2 = gray[ey:ey+eh, ex:ex+ew]
                roi_color2 = img[ey:ey+eh, ex:ex+ew]
                blur = cv2.GaussianBlur(roi_gray2,(5,5),10)
                erosion = cv2.erode(blur,kernel,iterations = 2)
                ret3,th3 = cv2.threshold(erosion,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                circles = cv2.HoughCircles(erosion,cv2.HOUGH_GRADIENT,4,200,param1=20,param2=150,minRadius=0,maxRadius=0)
                try:
                    for i in circles[0,:]:
                        if(i[2]>0 and i[2]<55):
                            cv2.circle(roi_color2,(i[0],i[1]),i[2],(0,0,255),1)
                            cv2.putText(img,"Pupil Pos:",(450,30), font, 1,(0,0,255),2,cv2.LINE_AA)
                            cv2.putText(img,"X "+str(int(i[0]))+" Y "+str(int(i[1])),(430,60), font, 1,(0,0,255),2,cv2.LINE_AA)
                            d = (i[2]/2.0)
                            dmm = 1/(25.4/d)
                            diameter.append(dmm)
                            cv2.putText(img,str('{0:.2f}'.format(dmm))+"mm",(10,60), font, 1,(0,0,255),2,cv2.LINE_AA)
                            cv2.circle(roi_color2,(i[0],i[1]),2,(0,0,255),3)
                            #cv2.imshow('erosion',erosion)
                except Exception as e:
                    pass
                
        else:
            if (blink==False):
                blink=True
                if blink==True:
                     cv2.putText(img,"Blink",(10,90), font, 1,(0,0,255),2,cv2.LINE_AA)
            a="Eye Close" 
            cv2.putText(img,a,(10,30), font, 1,(0,0,255),2,cv2.LINE_AA)
            
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    plt.plot(diameter)
    plt.ylabel('pupil Diameter')
    plt.show()        
    cap.release()
    cv2.destroyAllWindows()
except:
    plt.plot(diameter)
    plt.ylabel('pupil Diameter')
    plt.show()     

