import numpy as np
import matplotlib.pyplot as plt
import scipy


class Projection():
    """
        Use Projection class for finding Projection matrix, Camera Center, line, DR
        Varible:
            M             --> Projection Matrix
            camera_center --> Location of camera for future use

        Methods:
            projection_mat --> Find M based on .npy base point
            calc_line      --> Find projection line of input point
    """
    def __init__(self) -> None:
        # Variables
        self.M              = np.zeros((3,4), dtype=np.float32)
        self.camera_center  = np.zeros((1,3), dtype=np.float32)

    # Find projection matrix
    def projection_mat(self,data_path):
        # Reading Data from .npy file
        data = np.load(data_path)

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
        
        # Mul. of Transpose of A and A Find Projection Matrix with Eigen Vector that have min eigen-value
        A_ = A.T @ A
        eigenvalues, eigenvectors = np.linalg.eig(A_)
        self.M = eigenvectors[:,np.argmin(eigenvalues)]
        self.M = self.M / self.M[-1]
        self.M = np.reshape(self.M, (3,4))

        # Finding center of camera
        M_                        = self.M.T @ self.M 
        eigenvalues, eigenvectors = np.linalg.eig(M_)
        camera_center             = eigenvectors[:,np.argmin(eigenvalues)]
        self.camera_center        = camera_center[0:3] / camera_center[-1]
        print("Center of Camera : ", self.camera_center)

    
    # Find Line of projection point
    def calc_line(self,test_point):
        new_M = self.M 
        new_M[0,3] = new_M[0,3] - test_point[0]
        new_M[1,3] = new_M[1,3] - test_point[1]
        new_M[2,3] = new_M[2,3] - test_point[2]

        M_ = new_M.T @ new_M 
        eigenvalues, eigenvectors = np.linalg.eig(M_)
        sample_loc = eigenvectors[:,np.argmin(eigenvalues)]
        sample_loc = sample_loc[0:3] / sample_loc[-1]
        print("sample_loc       : ", sample_loc)

        # Line function based on Scale input variable
        line = lambda scale: self.camera_center + scale*(sample_loc-self.camera_center)

        # Secound output is DR
        return line, sample_loc-self.camera_center


# camera_8 = Projection()
# camera_8.projection_mat('./data/game_8.npy')
# line, DR = camera_8.calc_line(np.array([779,385, 1]))