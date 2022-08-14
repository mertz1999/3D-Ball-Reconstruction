from utils.projection import Projection
import pandas as pd
import numpy as np
import pickle
import cv2

# Load video, projection, csv for each of the inputs camera
def data_extract(data_path, video_path, csv_path, players_path):
    cap         = [] # video opencv object      for each camera
    total_frame = [] # Number of total frames   for each camera
    projection  = [] # Save Projection class    for each camera
    csv_data    = [] # Read csv data of ball    for each camera
    players_loc = [] # If player loc is include for each camera
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
    
    return cap, total_frame, projection, csv_data, players_loc

    

# Draw rectangle on image
def draw_rect_id(image, players_loc):
    # loc: x_min, y_min, x_max, y_max, id, class (0 is preson)
    for loc in  players_loc:
        if loc[5] == 0:
            cv2.rectangle(image, (int(loc[0]),int(loc[1])), (int(loc[2]),int(loc[3])), (0,255,0), 1)
            cv2.putText(image, '{}'.format(int(loc[4])), (int(loc[0])+5,int(loc[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)
        
    return image


# Assign more value to a point that has less distance to other points 
def repeat_min_dist(points_list, number_of_repeat = 5):
    for i in range(len(points_list)-1):
            dist = np.array([abs(sum((points_list[i]-point)**2))/3 for point in points_list[i+1:]])
            number_of_near = sum(dist < 5)
            for k in range(number_of_near*number_of_repeat): points_list.append(points_list[i])
    
    return points_list

# Make Output image
def output_img_creator(frames, court_img):
    output_img = np.zeros((576, 1024, 3), dtype=np.uint8)
    if len(frames) >= 4:
        output_img[0:288,0:512,:] = frames[0]
        output_img[0:288,512:,:]  = frames[1]
        output_img[288:,0:512,:]  = frames[2]
        output_img[288:,512:,:]   = frames[3]
        output_img[-150:,394:630,:] = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (236,150))

    # Output for 3 Camera
    elif len(frames) == 3:
        output_img[0:288,0:512,:] = frames[0]
        output_img[0:288,512:,:]  = frames[1]
        output_img[288:,0:512,:]  = frames[2]
        output_img[288:,512:,:]   = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (512,288))
    # Output for 2 Camera
    elif len(frames) == 2:
        output_img[0:288,0:512,:] = frames[0]
        output_img[0:288,512:,:]  = frames[1]
        output_img[288:,512:,:]   = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (512,288))
    elif len(frames) == 1:
        output_img = cv2.resize(frames[0], (1024, 576))
        output_img[-150:,394:630,:] = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (236,150))
    
    return output_img
