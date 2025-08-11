import cv2
import numpy as np
import json
import pickle
import argparse


# reset the robot
# with open('reset_robot.py') as f:
#     exec(f.read())
    
def nothing(self):
    pass

def get_camera_image():
    camera = cv2.VideoCapture(2)
    camera.set(cv2.CAP_PROP_EXPOSURE, -11)
    for idx in range(10):
        ret, img = camera.read()
    return img

def run_detection(args):
            
    image = get_camera_image() if args.load_image is None else cv2.imread(args.load_image)
            
    # Create a window
    cv2.namedWindow('image')
    f = 'R'
    s = 'G'
    t = 'B'
        
    # Create trackbars for color change
    cv2.createTrackbar('{}Min'.format(f), 'image', 0, 255, nothing)
    cv2.createTrackbar('{}Min'.format(s), 'image', 0, 255, nothing)
    cv2.createTrackbar('{}Min'.format(t), 'image', 0, 255, nothing)
    cv2.createTrackbar('{}Max'.format(f), 'image', 0, 255, nothing)
    cv2.createTrackbar('{}Max'.format(s), 'image', 0, 255, nothing)
    cv2.createTrackbar('{}Max'.format(t), 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('{}Max'.format(f), 'image', 255)
    cv2.setTrackbarPos('{}Max'.format(s), 'image', 255)
    cv2.setTrackbarPos('{}Max'.format(t), 'image', 255)

    # Initialize min/max values
    rMin = gMin = bMin = rMax = gMax = bMax = 0
    prMin = pgMin = pbMin = prMax = pgMax = pbMax = 0
    
    while True:

        # Get current positions of all trackbars
        rMin = cv2.getTrackbarPos('{}Min'.format(f), 'image')
        gMin = cv2.getTrackbarPos('{}Min'.format(s), 'image')
        bMin = cv2.getTrackbarPos('{}Min'.format(t), 'image')
        rMax = cv2.getTrackbarPos('{}Max'.format(f), 'image')
        gMax = cv2.getTrackbarPos('{}Max'.format(s), 'image')
        bMax = cv2.getTrackbarPos('{}Max'.format(t), 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([rMin, gMin, bMin])
        upper = np.array([rMax, gMax, bMax])

        # Convert to HSV format and color threshold
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.inRange(rgb, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((prMin != rMin) | (pgMin != gMin) | (pbMin != bMin) | (prMax != rMax) | (pgMax != gMax) | (pbMax != bMax) ):
            print("(rMin = %d , gMin = %d, bMin = %d), (rMax = %d , gMax = %d, bMax = %d)" % (rMin , gMin , bMin, rMax, gMax , bMax))
            phMin = rMin
            psMin = gMin
            pvMin = bMin
            phMax = rMax
            psMax = gMax
            pvMax = bMax

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return (rMin, gMin, bMin), (rMax, gMax, bMax)


# choose your save name
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, type=str, default=None)
parser.add_argument('--load-image', type=str, default=None)
args = parser.parse_args()

# detect thresholds and save them
colors_thresholds = run_detection(args)
json.dump(colors_thresholds, open("./config/" + args.name + ".json", "w") )

# print what you are saving
print(json.load(open("./config/" + args.name + ".json", "r")))