#Author: Neeyati Ajmera
#CS2520
#Python Project 

"""
Face and Eye detection project: 
This Project utilizes OpenCV to build a face and eye detection program. 
It's primarily used to track faces in a live video. 
The objective of this Python project is to learn about OpenCV. 
OpenCV is a library of programming functions mainly aimed at real-time computer vision.

This program uses the webcam as the video source in order to detect faces and eyes. Upon detection, it draws a green rectangle around it
 
"""

#Import cv2 which is the second version of OpenCV library
import cv2

"""
OpenCV comes with many pre-trained classifiers. I have used the haarcascade_frontalface as well as haarcascade_eye classifiers that come as a xml file
A face cascade and an eye cascade are created using that classifiers
"""

#Create a face cascade using haarcascade frontalface classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Create an eye cascade using haarcascade eye classifier
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 



#Open the default web camera 
cap = cv2.VideoCapture(0) 

#Create a loop to read frames from the video captured. It reads one frame per loop iteration
while True:
    
    """
    read() function reads one frame from the video captured and returns the frame read. I then converted the frame captured
    to a gray-scale frame
    """
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    
    """
    Detect faces in the gray-scale version of the frame captured:
    I have used the detectMultiscale module of this classifier. 
    This function will return a rectangle with coordinates(x,y,w,h) around the detected face.
    
    """
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        )
    """
    rectangle() function in cv2:
        
    frame = frame captured by webcam to be used for facial detection
    point 1 = (x,y) = vertex of the rectangle
    point 2 = (x+w, y+h) = vertex of the rectangle opposite to point 1
    color = green = (0,255,0)
    thickness = 2
    
    """
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w , y+h), (0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray) 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 

    """
    imshow() function has two parameters, frame = the rgb frame captured from the video and the name of the
    window in which the frame would be displayed
    """
    cv2.imshow("facial detection demo",frame)
    
    """
    waitKey(1) function:
    It will wait for keyPress for just 1 millisecond and after that it will continue to refresh and read frame from your webcam
    It returns a 32-bit integer corresponding to the pressed key
    
    ord() function:
    returns a unicode point of the character parameter
    
    When 'q' key is pressed, the while loop is exited. 
    
    """
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()