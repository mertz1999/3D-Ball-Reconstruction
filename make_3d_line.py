import numpy as np
import matplotlib.pyplot as plt
import scipy



# Reading Data from .npy file
data = np.load('./data/game_8.npy')

# Simple plotting
fig = plt.figure()
ax  = plt.axes(projection='3d')
ax.scatter3D(data[:,0],data[:,1],data[:,2], cmap='Greens');
# plt.show()

# Make A Matrix
number_of_points = len(data)
A                = np.zeros((number_of_points*2,12), dtype=np.float64)
for i in range(number_of_points):
    temp  = data[i,0:3] * -1 * data[i,3]
    temp2 = data[i,0:3] * -1 * data[i,4]

    A[i*2, :]   = np.concatenate((np.array([data[i,0], data[i,1], data[i,2], 1, 0, 0, 0, 0]),
                                  temp,
                                  np.array([-1*data[i,3]]),
                                ), axis=0)

    A[i*2+1, :] = np.concatenate((np.array([0, 0, 0, 0, data[i,0], data[i,1], data[i,2], 1]),
                                  temp2,
                                  np.array([-1*data[i,4]]),
                                ), axis=0)

# print(A)
# print(np.array_str(A, precision=1, suppress_small=True))

# Mul. of Transpose of A and A Find Projection Matrix with Eigen Vector that have min eigen-value
A_ = A.T @ A
eigenvalues, eigenvectors = np.linalg.eig(A_)
M = eigenvectors[:,np.argmin(eigenvalues)]
M = M / M[-1]
M = np.reshape(M, (3,4))

# Finding center of camera
M_ = M.T @ M 
eigenvalues, eigenvectors = np.linalg.eig(M_)
camera_center = eigenvectors[:,np.argmin(eigenvalues)]
camera_center = camera_center[0:3] / camera_center[-1]
print("Center of Camera: ", camera_center)


ax.scatter3D(camera_center[0],camera_center[1],camera_center[2], cmap='Greens');
plt.show()