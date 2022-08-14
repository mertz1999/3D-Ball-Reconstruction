"""
This file is used to extract 3D point of ball and players by multiple
cameras (At least 2 camera).
Input of this file is a .json file that contain .mp4, .csv, .npy, .pkl.

for using this file only assign .json path to --config option.


"""


# Import all packages
from utils.utils import data_extract, draw_rect_id, repeat_min_dist, output_img_creator
from utils.projection import Projection, nearest_point, person_loc
from utils.make_court import Court
import numpy as np
import argparse
import json
import cv2 

# Args parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file', required=True)
parser.add_argument('--shift', type=int, help='number of shifting', default=2)
parser.add_argument('--show', action="store_true", help='Show output? ')
parser.add_argument('--start', type=int, help='Start frame index', default=20)
parser.add_argument('--stop', type=int, help='stop frame index', default=-1)
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


WIDTH        = 512
HEIGHT       = 288
START        = args.start
STOP         = args.stop


# Read Data for each of camera (.mp4, .npy, .csv, .pkl) and save them as list
cap, total_frame, projection, csv_data, players_loc = data_extract(data_path, video_path, csv_path, players_path)

# make ready to write output video
fourcc  = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
out_vid = cv2.VideoWriter(output, fourcc, int(cap[0].get(cv2.CAP_PROP_FPS)), (1024,576),True)

# Court maker init (Use this for make a court image with player, line and ball location drawing)
court_class = Court()

# Read data frame by frame
last = min(total_frame) if STOP > min(total_frame) or STOP==-1 else STOP
for idx in range(START,last):
    print("frame: ",idx)

    # Get empty cout image for write on it
    court_img = court_class.court_image.copy()

    # --- Saving informations
    """
        Part 1: Saving informations
        
        This part is used to save some detailed information like
            1. frames for each camera
            2. Save player data that belong to this camera frame
            3. Save prev. player location to p_players variable
            4. Save all line that belong to ball in each frame
    """
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

        # Draw and save player location 
        if players_loc[i] != -1:
            # Save player data for each frame camera
            plr_loc.append(players_loc[i][idx])   # Save This frame player data

            # Save 5 or more prev. player data to make moving average in future
            for prev_idx in range(1,5):
                if prev_idx == 1:
                    p_players[i] = {idx-1 : players_loc[i][idx-1]}
                else:
                    p_players[i][idx-prev_idx] = players_loc[i][idx-prev_idx]
            
            # Draw rectangle on playera
            image = draw_rect_id(image, players_loc[i][idx])
        else:
            plr_loc.append([])

        # Change image size if we have multiple camera
        if len(data_path) != 1:
            image = cv2.resize(image, (512, 288))

        # Save frames
        frames.append(image)

        # Make projected line based on cx,cy
        cx = int(csv_data[i]['X'][csv_data[i].Frame == idx].iloc[0])
        cy = int(csv_data[i]['Y'][csv_data[i].Frame == idx].iloc[0])
        if cx == 0.0 and cy == 0.0:
            lines.append(-1) # If visibility of ball being False line = -1
        else:
            lines.append(projection[i].calc_line(np.array([cx,cy, 1])))

    # --- Find Point in 3D (1-by-1)
    """
        Part 2: Find point in 3D (1-by-1)
        
        This part is used to find 3D point based on 1-by-1 cameras and draw on court
            1. Find nearst point with each 1-1 camera pairs for ex. 1-2 1-3 1-4 2-2 2-4 3-4
            2. With repeat_min_dist function repeat a value that hase lower distance to other values in a list (points in 3D)
            3. Draw line and ball location (if it finded)
    """
    total_detected = np.array([0.0,0.0,0.0]) # sum all detected point and save them in total_detected variable
    points_list = []
    for i in range(len(data_path)-1):
        for j in range(i+1,len(data_path)):
            if lines[i] != -1 and lines[j] != -1:
                points_list.append(nearest_point(lines[i],lines[j]))
            
    # Assign more value to a point that has less distance to other points 
    if len(points_list) >= 3:
        points_list = repeat_min_dist(points_list, number_of_repeat = 5)
    

    # Draw line and ball point in 2D format on court image
    if len(points_list) != 0 or len(data_path) == 1:
        # If we have only one cemra
        if len(data_path) != 1:
            detected_point = sum(points_list) / len(points_list)
            court_img      = court_class.make_image(detected_point[0], detected_point[1])
        else:
            detected_point = [0,0,-1.0]
        
        # Draw line by 
        for i in range(len(data_path)):
            court_img = court_class.make_line(lines[i],court_img)
    else:
        detected_point = [0,0,-1.0]

    # --- Make person locations:
    """
        Part 3: Make person location with Moving-Average
        
        This part is used to make an aveage of player positions with a specific id in prev. frames
     
    """
    for i, locs in enumerate(plr_loc):
        if locs != []:
            for loc in locs:
                if loc[5] == 0.0:
                    # find line and then xyz position of bottom of player
                    mean_x = (loc[0] + loc[2])/2
                    xyz = person_loc(projection[i].calc_line(np.array([mean_x, loc[3], 1])), stop=200)

                    # Check if ID of this player is in prev frames data
                    prev_find = 0
                    if len(p_players) != 0:
                        p_data    = p_players[i]
                        for p_index in p_data:
                            list_ids  = [j[4] for j in p_data[p_index]] # List of ids
                            if loc[4] in list_ids:                      # Check if player id exist on one of the prev. frame data
                                prev_find += 1                          # find number of matched players
                                sl_idx = list_ids.index(loc[4])       
                                mean_x = (p_data[p_index][sl_idx][0] + p_data[p_index][sl_idx][2])/2
                                xyz    = xyz + person_loc(projection[i].calc_line(np.array([mean_x, p_data[p_index][sl_idx][3], 1])), stop=200) 

                    xyz = xyz / (prev_find+1)    
                    court_img = court_class.make_image(xyz[0], xyz[1], court_img, status='player')

    

    # make output image
    """
        Part 4: Make output image
             
    """
    output_img = output_img_creator(frames, court_img)

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