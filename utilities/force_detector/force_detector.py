import time
import serial
import socket
import json


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
    port = "/dev/rfcomm0"
    HOST = '127.0.0.1'
    PORT = 38090
    
    ser = serial.Serial(port, 9600, timeout=1)
    print("Connected to Arduino")
    
    ser.write(b'q')
    time.sleep(0.5)
    force_triggered = False
    
    safety_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("[CLIENT] Connecting to safety server...")
    safety_socket.connect((HOST,PORT))
    safety_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    print("[CLIENT] Connected.")
    
    last_safety_send = 0
    send_interval = 0.1
    while True:
        try:
            arduino_input = ser.readline().decode().strip()
            print(arduino_input)
            if arduino_input == '1':
                print("FORCE TRIGGERED")
                force_triggered = True
            
            # Sending stop signal to run.py
            if force_triggered:
                if time.time() - last_safety_send > send_interval:  
                    try:
                        data = {"safety": "force_triggered"} 
                        if talk2server(safety_socket, data): 
                            last_safety_send = time.time() 
                    
                    except Exception as e: 
                        print("[CLIENT] Failed to send force safety alert:", e) 
        except Exception as e:
            print(e)
            print('something went wrong')
            exit(0)
            
        if time.time() - last_safety_send > send_interval:  
            try:
                data = {"safety": "safe"} 
                if talk2server(safety_socket, data): 
                    last_safety_send = time.time()
                    
            except Exception as e: 
                    print("[CLIENT] Failed to send force safety alert:", e)
                    
                    

if __name__ == '__main__':
    main()