# -*- coding: utf-8 -*-
import cv2
import sys

def camcam():
    cap=cv2.VideoCapture(0)
    qImg=None
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            print('No NO No!!')
            break
        h,w=frame.shape[:2]
        #화면에 책 인식 영역 표시
        left=w//3
        right=(w//3)*2
        top=(h//2)-(h//3)
        bottom=(h//2)+(h//3)
        cv2.rectangle(frame,(left,top),(right,bottom),(255,255,255),3)
    
        flip=cv2.flip(frame,1)
        cv2.imshow('Boo',flip)
        key=cv2.waitKey(10)
        if key==ord(' '): #space bar 입력 시 마스크 안 부분 영상 저장
            qImg=frame[top:bottom,left:right]
            cv2.destroyAllWindows()
            return qImg
            break
        elif key==27:
            cv2.destroyAllWindows()
            sys.exit()
    else:
        print('No Cam')
    cap.release()
    
    