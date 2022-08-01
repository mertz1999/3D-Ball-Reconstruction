from cmath import pi
from utils.projection import Projection, nearest_point
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import pandas as pd
from utils.make_court import Court


data_path = ['./data/1_p1.npy', './data/2_p1.npy']

video_path = [
    './games/real/1_1_predicted_improved.mp4',
    './games/real/2_1_predicted_improved.mp4'
    ]

csv_path = [
    './games/real/1_1_predicted.csv',
    './games/real/2_1_predicted.csv'
]


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
out_vid = cv2.VideoWriter('./output/1.mp4', fourcc, int(cap[0].get(cv2.CAP_PROP_FPS)), (1024, 576),True)


# Read data
court_class = Court()
for idx in range(2,min(total_frame)):
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
            flag = True 

        lines.append(projection[i].calc_line(np.array([cx,cy, 1])))


    # Find Point in 3D
    if flag != True:
        detected_point = nearest_point(lines[0],lines[1])
        court_img = court_class.make_image(detected_point[0], detected_point[1])
        # court_img = court_class.make_line(lines[0],lines[1])
    else:
        detected_point = [0,0,-1.0]
    

    # make output image
    output_img = np.zeros((576,1024,3), dtype=np.uint8)

    output_img[0:288,0:512,:] = frames[0]
    output_img[288:,512:,:] = frames[1]
    output_img[0:288,512:,:] = cv2.resize(cv2.rotate(court_img,cv2.ROTATE_90_CLOCKWISE), (512,288))

    output_img = cv2.putText(output_img, 'Height: {:.2f}'.format(detected_point[2]), (520,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    out_vid.write(output_img)

    # cv2.imshow('result', output_img)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

out_vid.release()