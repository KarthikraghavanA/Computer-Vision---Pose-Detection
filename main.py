import cv2
import numpy as np
import mediapipe as mnp

mpose = mnp.solutions.pose
pose = mpose.Pose() # Detects pose

mdraw = mnp.solutions.drawing_utils


vid1 = cv2.VideoCapture('Pose3.mp4')

drawspec1 = mdraw.DrawingSpec(thickness=2, circle_radius=3, color=(0,0,255)) # thickness of points and line
drawspec2 = mdraw.DrawingSpec(thickness=2, circle_radius=3, color=(0,255,0)) # thickness of points and line


while True:
    success, img = vid1.read()
    img = cv2.resize(img, (400,400))
    results = pose.process(img)
    mdraw.draw_landmarks(img,results.pose_landmarks, mpose.POSE_CONNECTIONS,drawspec1, drawspec2)

    h,w,c = img.shape
    blankImg = np.zeros([h,w,c])
    blankImg.fill(255)
    mdraw.draw_landmarks(blankImg,results.pose_landmarks, mpose.POSE_CONNECTIONS,drawspec1, drawspec2)


    cv2.imshow('PoseDetection', img)
    cv2.imshow('Blank Image', blankImg)
    cv2.waitKey(1)
