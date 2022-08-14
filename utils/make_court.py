import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

# Court Class to return ball or player on court image
class Court():
    """
        Use this class to return a court image with circle points on it or drawing lines
    """
    def __init__(self, court_path='./inc/court.jpg'):
        self.court_image = cv2.imread(court_path)
        # some point in 3D vollyball court and 2D vollyball court (./inc/cout.jpg)
        pts1 = np.float32([[0, 0],[0, -18],[9, -18]])
        pts2 = np.float32([[64, 577],[64, 50],[344, 50]])

        # Find transformation between pts1 and pts2
        self.M = cv2.getAffineTransform(pts1, pts2)

    # Make output image
    def make_image(self, cx,cy, image=np.array([]), status='ball'): # status: 'ball', 'player'
        point = self.M @ (np.array([cx,cy,1]).T)
        if len(image) == 0:
            court_copy = self.court_image.copy()
        else:
            court_copy = image
        if status == 'ball':
            court_copy = cv2.circle(court_copy, (int(point[0]),int(point[1])), 10, (0,0,0), -1)
        elif status == 'player':
            court_copy = cv2.circle(court_copy, (int(point[0]),int(point[1])), 10, (0,0,255), -1, cv2.LINE_AA)

        return court_copy
    
    def make_line(self, line1, image=np.array([])):
        if len(image) == 0:
            image = self.court_image.copy()
        try:
            p0 = line1(0)
            p1 = line1(50)

            p0 = self.M @ (np.array([p0[0],p0[1],1]).T)
            p1 = self.M @ (np.array([p1[0],p1[1],1]).T)
            image = cv2.line(image, (int(p0[0]),int(p0[1])), (int(p1[0]),int(p1[1])), (255,0,0), 1)
        except:
            pass

        return image

