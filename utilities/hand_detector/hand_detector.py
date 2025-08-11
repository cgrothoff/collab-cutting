import time

import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2 
from mediapipe import solutions 
import cv2

import serial
import socket
import json

port = "/dev/rfcomm0"

HOST = '127.0.0.1'
PORT = 38088


class Camera(object):
    def __init__(self):
        self.camera = cv2.VideoCapture(-1) 
        self.camera.set(cv2.CAP_PROP_EXPOSURE, -11)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.fps = cv2.CAP_PROP_FPS
        self.t = 0
        
    def read(self):
        if self.camera.isOpened():
            for idx in range(1):
                ret, img = self.camera.read()
                if not ret:
                    print('No Frame Returned.')
            self.t += 1 / self.fps
        else:
            print('Camera could not be opened')
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
                                   
def check_safety_cropped(detection, ser, last_color):
    hand_landmarks_list = detection.hand_landmarks
    handedness_list = detection.handedness
    num_red = 0
    num_yellow = 0
    num_green = 0

    # Bounding Boxes
    red_x_start, red_y_start = (102.0)/433.0 , 40.0/270.0
    red_x_end, red_y_end = 433.0/433.0 , 270.0/270.0
    
    for idx in range(len(detection.hand_landmarks)):  
        hand_landmarks = detection.hand_landmarks[idx] 
            
        for landmark in hand_landmarks: 
            x, y = landmark.x, landmark.y  
            #print(f"Landmark: x={x}, y={y}")
            
            if (x > red_x_start and x < red_x_end) and (y > red_y_start and y < red_y_end):
                num_red = num_red + 1    
            else:
                num_yellow = num_yellow + 1
                
    # if len(detection.hand_landmarks) != 0:
    #     print("%0.2f%% Red | %0.2f%% Yellow" % (num_red/21.0*100, num_yellow/21.0*100))

    # 1 - Red | 2 - Yellow | 3 - Green
    if num_red > 0:
        color = '1'
    elif num_yellow > 0:
        color = '2'
    else:
        color = '3'
    
    # Only send color code when color changes
    if last_color != color:
        tic = time.time()
        ser.write(color.encode())
        toc = time.time() - tic
        print('Time to turn lights ', toc)
        
    return color

def talk2server(safety_socket, data): 
    connection_failed = False
    
    try: 
        msg = json.dumps(data) + '\n' 
        safety_socket.sendall(msg.encode('utf-8')) 
        return True
    except BrokenPipeError: 
        if not connection_failed:
            print("[CLIENT] Socket closed by server. Stopping communication.") 
            connection_failed = True
            exit(0)
        return None 
    except Exception as e: 
        print("[CLIENT] Failed to send data:", e) 
        return None 

def main():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
                                    running_mode=VisionRunningMode.VIDEO,
                                    num_hands=2, min_tracking_confidence=0.2, min_hand_presence_confidence=0.12)
    
    cam = Camera()
    

    # Set up Arduino
    ser = serial.Serial(port, 9600, timeout=1)
    print("Connected to Arduino")
    new_color = '3'
    
    # Reset command for force sensor
    ser.write(b'q')
    time.sleep(0.5)
    
    # Set up socket connection
    safety_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    print("[CLIENT] Connecting to safety server...")
    safety_socket.connect((HOST,PORT))
    safety_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    print("[CLIENT] Connected.")
    
    last_safety_send = 0
    send_interval = 0.1
    
    with HandLandmarker.create_from_options(options) as lm:
        while True:
            img, t = cam.read()
            
            # If camera returns frame
            if img is None:
                print('No Frame Returned.')
                continue
            
            # Cropping image to area of interest
            # x_start, y_start = 163, 187
            # x_end,y_end = 450, 408
            x_start, y_start = 120, 120
            x_end, y_end = 510, 410
            cropped_img = img[y_start:y_end, x_start:x_end].astype(np.uint8)
            
            # Creates compatible image
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_img)
            
            # Landmarker detects hand from video
            detection = lm.detect_for_video(mp_img, t)
            
            # Draws landmarks where the detector warned
            annotated_img = draw_landmark(mp_img.numpy_view(), detection)
            
            # Check for hand in workspace
            new_color = check_safety_cropped(detection, ser, new_color)
            
            cv2.imshow('Hand Detector', annotated_img)
                        
            # Sending stop signal to run.py
            if new_color == '1': 
                if time.time() - last_safety_send > send_interval: 
                    data = {"safety": "hand_detected"} 
                    if talk2server(safety_socket, data): 
                        last_safety_send = time.time() 
            # if new_color == '1':
            #     message = "HAND DETECTED"
            #     try: 
            #         data = {"safety": "hand_detected"} 
            #         resp = talk2server(data) 

            #         print("[CLIENT] Response from server:", resp) 

            #     except Exception as e: 
            #         print("[CLIENT] Failed to send hand safety alert:", e) 
            # else: 
            #     message = "SAFE"  
                
            # Check for force sensor failure
            # try:
            #     arduino_input = ser.readline().decode().strip()
            #     print(arduino_input)
            #     if arduino_input == '1':
            #         print("FORCE TRIGGERED")
            #         force_triggered = True
                
            #     # Sending stop signal to run.py
            #     if force_triggered:
            #         if time.time() - last_safety_send > send_interval:  
            #             try:
            #                 data = {"safety": "force_triggered"} 
            #                 if talk2server(safety_socket, data): 
            #                     last_safety_send = time.time() 
                        
            #             except Exception as e: 
            #                 print("[CLIENT] Failed to send force safety alert:", e) 
            # except:
            #     print('something went wrong')
            #     exit(0)
            
            if time.time() - last_safety_send > send_interval:  
                try:
                    data = {"safety": "safe"} 
                    if talk2server(safety_socket, data): 
                        last_safety_send = time.time() 
                
                except Exception as e: 
                    print("[CLIENT] Failed to send force safety alert:", e) 
            
            # Close out
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break
               
            
            
if __name__ == '__main__':
    main()