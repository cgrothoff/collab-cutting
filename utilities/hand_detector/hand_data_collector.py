import time

import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2 
from mediapipe import solutions 
import cv2

class Camera(object):
    def __init__(self):
        self.camera = cv2.VideoCapture(1)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, -11)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.fps = cv2.CAP_PROP_FPS
        self.t = 0
        
    def read(self):
        for idx in range(10):
            ret, img = self.camera.read()
        self.t += 1 / self.fps
        return img, int(self.t * 1000.)


def draw_landmark(img, detection):
    hand_landmarks_list = detection.hand_landmarks
    handedness_list = detection.handedness
    annotated_img = np.copy(img)
    
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, 
                                              y=landmark.y, z=landmark.z) for landmark in 
                                              hand_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_img,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    return annotated_img 

def get_landmarks(img, detection):
    hand_landmarks_list = detection.hand_landmarks()
        

def main():    
    # 21 hand landmarks
    # Each landmark has 3 coordinates: x, y, z in relation to screen
    # World landmarks have 3 coordinates: x, y, z in meters in world coordinates
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
                                    running_mode=VisionRunningMode.VIDEO,
                                    num_hands=2, min_tracking_confidence=0.4, min_hand_presence_confidence=0.4)
    
    cam = Camera()
    
    with HandLandmarker.create_from_options(options) as lm:
        while True:
            img, t = cam.read()

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            detection = lm.detect_for_video(mp_img, t)
            annotated_img = draw_landmark(mp_img.numpy_view(), detection)
            
            cv2.imshow('detector', annotated_img)
            if (cv2.waitKey(25) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break
            
if __name__ == '__main__':
    main()