from utils.projection import Projection, nearest_point
from utils.make_court import Court
import matplotlib.pyplot as plt
from cmath import pi
import pandas as pd
import numpy as np
import argparse
import json
import cv2 

# Loading json data
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file', required=True)
args = parser.parse_args()


f = open(args.config)
data_readed = json.load(f)


output     = data_readed['output']
data_path  = data_readed['data_path']
video_path = data_readed['video_path']
csv_path   = data_readed['csv_path']


# Load video, projection, csv for prodiction
cap         = []
total_frame = []
projection  = []
csv_data    = []

for i in range(len(data_path)):
    # For Video
    cap_temp    = cv2.VideoCapture(video_path[i])
    cap.append(cap_temp)
    total_frame.append(int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Projection
    proj_temp = Projection()
    proj_temp.projection_mat(data_path[i])
    projection.append(proj_temp)

    # CSV data from models
    csv_data.append(pd.read_csv(csv_path[i]))

# Output Video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
out_vid = cv2.VideoWriter(output, fourcc, int(cap[0].get(cv2.CAP_PROP_FPS)), (1024, 576),True)


# Read data
court_class = Court()
for idx in range(2,min(total_frame)//2):
    print(idx)
    court_img = court_class.court_image
    frames = []
    lines  = []
    
    flag = False
    for i in range(len(data_path)):
        # Read Each frame
        cap[i].set(cv2.CAP_PROP_POS_FRAMES, idx-2)
        _, image = cap[i].read()
        frames.append(image)

        # Save min distance point
        cx = int(csv_data[i]['X'][csv_data[i].Frame == idx].iloc[0])
        cy = int(csv_data[i]['Y'][csv_data[i].Frame == idx].iloc[0])
        if cx == 0.0 and cy == 0.0:
            lines.append(-1)
        else:
            lines.append(projection[i].calc_line(np.array([cx,cy, 1])))

    # Find Point in 3D
    total_detected = np.array([0.0,0.0,0.0])
    len_points = 0
    for i in range(len(data_path)-1):
        for j in range(i+1,len(data_path)):
            if lines[i] != -1 and lines[j] != -1:
                len_points += 1
                detected_point = nearest_point(lines[i],lines[j])
                total_detected += detected_point
  
    detected_point = total_detected / len_points

    if sum(total_detected) != 0.0:
        court_img = court_class.make_image(detected_point[0], detected_point[1])
        for i in range(len(data_path)):
            court_img = court_class.make_line(lines[i],court_img)
    else:
        detected_point = [0,0,-1.0]
    

    # make output image
    output_img = np.zeros((576,1024,3), dtype=np.uint8)
    # Output for 4 camera
    if len(data_path) == 4:
        output_img[0:288,0:512,:] = frames[0]
        output_img[0:288,512:,:]  = frames[1]
        output_img[288:,0:512,:]  = frames[2]
        output_img[288:,512:,:]   = frames[3]
        output_img[-150:,394:630,:] = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (236,150))

        output_img = cv2.putText(output_img, 'Height : {:.2f}'.format(detected_point[2]), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
        output_img = cv2.putText(output_img, 'X      : {:.2f}'.format(detected_point[0]), (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
        output_img = cv2.putText(output_img, 'Y      : {:.2f}'.format(detected_point[1]), (25,75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
    # Output for 3 Camera
    elif len(data_path) == 3:
        output_img[0:288,0:512,:] = frames[0]
        output_img[0:288,512:,:]  = frames[1]
        output_img[288:,0:512,:]  = frames[2]
        output_img[288:,512:,:]   = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (512,288))
    # Output for 2 Camera
    elif len(data_path) == 2:
        output_img[0:288,0:512,:] = frames[0]
        output_img[0:288,512:,:]  = frames[1]
        output_img[288:,512:,:]   = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (512,288))


    out_vid.write(output_img)

    # cv2.imshow('result', output_img)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

out_vid.release()