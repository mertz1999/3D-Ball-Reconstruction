from utils.projection import Projection, nearest_point
import numpy as np
import matplotlib.pyplot as plt


data_path = ['./data/1_p1.npy', './data/2_p1.npy', './data/4_p1.npy']



# Load Volley cordinates
data = np.load('./data/volley_cords.npy')

# Plotting Volley cordinate Points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(data[:,1], data[:,0], data[:,2], marker='^')
# plt.show()

for i in data_path:
    test_camera = Projection()
    test_camera.projection_mat(i)
    camera_loc = test_camera.camera_center

    ax.scatter(camera_loc[1], camera_loc[0], camera_loc[2], marker='o')
plt.show()
