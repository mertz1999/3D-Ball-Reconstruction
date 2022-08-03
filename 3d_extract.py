from ast import arg
from utils.projection import Projection, nearest_point, person_loc
from utils.make_court import Court
import matplotlib.pyplot as plt
from cmath import pi
import pandas as pd
import numpy as np
import argparse
import pickle
import json
import cv2 

# Args parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file', required=True)
parser.add_argument('--shift', type=int, help='number of shifting', default=2)
parser.add_argument('--show', action="store_true", help='Show output? ')
args = parser.parse_args()

# Load Json file
f = open(args.config)
data_readed = json.load(f)

# Save information from json file
output       = data_readed['output']
data_path    = data_readed['data_path']
video_path   = data_readed['video_path']
csv_path     = data_readed['csv_path']
players_path = data_readed['players_path']




# Load video, projection, csv for prodiction
cap         = []
total_frame = []
projection  = []
csv_data    = []
players_loc = []
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

    # Player
    if players_path[i] != "":
        with open(players_path[i], 'rb') as file:
            players_loc.append(pickle.load(file))
    else:
        players_loc.append(-1)

# Output Video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
out_vid = cv2.VideoWriter(output, fourcc, int(cap[0].get(cv2.CAP_PROP_FPS)), (1024, 576),True)

# Court maker init
court_class = Court()

# Read data frame by frame
for idx in range(20,min(total_frame)//2):
    print(idx)
    court_img = court_class.court_image.copy()

    # --- Saved informations
    frames     = [] # Save frames of all videos
    lines      = [] # Save projected line based on cx,cy ig
    plr_loc    = [] # all players location of this frame index
    p_players  = {} # prev. player location
    pp_players = {} # double prev. player location
    flag = False
    for i in range(len(data_path)):
        # Read frame from cemras
        cap[i].set(cv2.CAP_PROP_POS_FRAMES, idx-args.shift)
        _, image = cap[i].read()

        # Draw player location on frame
        if players_loc[i] != -1:
            plr_loc.append(players_loc[i][idx])   # Save This frame player data
            for prev_idx in range(1,10):
                if prev_idx == 1:
                    p_players[i] = {idx-1 : players_loc[i][idx-1]}
                else:
                    p_players[i][idx-prev_idx] = players_loc[i][idx-prev_idx]
            for loc in  players_loc[i][idx]:
                if loc[5] == 0:
                    # cv2.rectangle(image, (int(loc[0]),int(loc[1])), (int(loc[2]),int(loc[3])), (0,255,0), 1)
                    mean_x = (loc[0] + loc[2])/2
                    cv2.line(image, (int(mean_x)-10,int(loc[3])), (int(mean_x)+10,int(loc[3])), (0,255,0), 1)
                    cv2.putText(image, '{}'.format(int(loc[4])), (int(mean_x),int(loc[3])+8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)
        
        else:
            plr_loc.append([])
        frames.append(image)


        # Make projected line based on cx,cy
        cx = int(csv_data[i]['X'][csv_data[i].Frame == idx].iloc[0])
        cy = int(csv_data[i]['Y'][csv_data[i].Frame == idx].iloc[0])
        if cx == 0.0 and cy == 0.0:
            lines.append(-1) # If visibility of ball being False line = -1
        else:
            lines.append(projection[i].calc_line(np.array([cx,cy, 1])))

    # --- Find Point in 3D (2-by-2)
    total_detected = np.array([0.0,0.0,0.0]) # sum all detected point and save them in total_detected variable
    len_points = 0                           # Number of points
    len_coeff = 0
    points_list = []
    for i in range(len(data_path)-1):
        for j in range(i+1,len(data_path)):
            if lines[i] != -1 and lines[j] != -1:
                len_points += 1
                detected_point = nearest_point(lines[i],lines[j])
                points_list.append(detected_point)

    # Best best point
    if len_points >= 3:
        for i in range(len_points-1):
            dist = np.array([abs(sum((points_list[i]-point)**2))/3 for point in points_list[i+1:]])
            number_of_near = sum(dist < 5)
            for k in range(number_of_near*5): points_list.append(points_list[i])
    

    # Draw line and ball point in 2D format on court image
    if len(points_list) != 0 or len(data_path) == 1:
        if len(data_path) != 1:
            detected_point = sum(points_list) / len(points_list)
            court_img = court_class.make_image(detected_point[0], detected_point[1])
        else:
            detected_point = [0,0,-1.0]
        for i in range(len(data_path)):
            court_img = court_class.make_line(lines[i],court_img)
    else:
        detected_point = [0,0,-1.0]

    # Make person locations:
    for i, locs in enumerate(plr_loc):
        if locs != []:
            for loc in locs:
                if loc[5] == 0:
                    # Now point
                    mean_x = (loc[0] + loc[2])/2
                    xyz = person_loc(projection[i].calc_line(np.array([mean_x, loc[3], 1])))

                    prev_find = 0
                    p_data    = p_players[i]
                    for p_index in p_data:
                        list_ids  = [j[4] for j in p_data[p_index]]
                        if loc[4] in list_ids:
                            prev_find += 1
                            sl_idx = list_ids.index(loc[4])
                            mean_x = (p_data[p_index][sl_idx][0] + p_data[p_index][sl_idx][2])/2
                            xyz    = xyz + person_loc(projection[i].calc_line(np.array([mean_x, p_data[p_index][sl_idx][3], 1]))) 

                    xyz = xyz / (prev_find+1)                    
                    court_img = court_class.make_image(xyz[0], xyz[1], court_img, status='player')

    

    # make output image
    output_img = np.zeros((576,1024,3), dtype=np.uint8)
    # Output for 4 camera
    if len(data_path) >= 4:
        output_img[0:288,0:512,:] = frames[0]
        output_img[0:288,512:,:]  = frames[1]
        output_img[288:,0:512,:]  = frames[2]
        output_img[288:,512:,:]   = frames[3]
        output_img[-150:,394:630,:] = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (236,150))

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
    elif len(data_path) == 1:
        output_img = cv2.resize(frames[0], (1024,576))
        output_img[-150:,394:630,:] = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (236,150))

    output_img = cv2.putText(output_img, 'Height : {:.2f}'.format(detected_point[2]), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
    output_img = cv2.putText(output_img, 'X      : {:.2f}'.format(detected_point[0]), (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
    output_img = cv2.putText(output_img, 'Y      : {:.2f}'.format(detected_point[1]), (25,75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)


    if args.show:
        cv2.imshow('result', output_img)
        # cv2.waitKey(0)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        out_vid.write(output_img)
out_vid.release()