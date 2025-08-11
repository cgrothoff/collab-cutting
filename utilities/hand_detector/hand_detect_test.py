import time

import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2 
from mediapipe import solutions 
import cv2

def check_safety(detection):
    hand_landmarks_list = detection.hand_landmarks
    handedness_list = detection.handedness
    
    for idx in range(len(detection.hand_landmarks)):  
        hand_landmarks = detection.hand_landmarks[idx] 
           
        for landmark in hand_landmarks: 
            x, y = landmark.x, landmark.y  
            print(f"Landmark: x={x}, y={y}")
            
            if x > 218 & x < 491:
                print("RED")
            elif x > 218-30 & x > 491+30:
                print("YELLOW")
            else:
                print("safe")


def main():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Flush Frames
    for _ in range(5):
        cam.read()
    
    while cam.isOpened():
            success, frame = cam.read()
            
            # If camera returns frame
            if not success:
                print('Warning: No frame returned from camera.')
                continue
            
            # table - red
            cv2.rectangle(frame, (218, 68), (491, 480), (0, 0, 0), 3)
            
            #safety box - yellow?
            cv2.rectangle(frame, (218-30, 20), (491+30, 480), (0, 0, 0), 3)
            
            cv2.imshow('hand detector', frame)
               
            # Close out
            if (cv2.waitKey(25) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break
            
            # 2/12 Notes:
            # Even just testing if the camera is working, it takes 
            # a long time for the camera to start, and then will 
            # freeze and give up 30 seconds later
    cam.release()    
            
            
if __name__ == '__main__':
    main()