import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

# Court Class to return ball or player on court image
class Court():
    """
        Use this class to return a court image with circle points on it.
    """
    def __init__(self, court_path='./inc/court.jpg'):
        self.court_image = cv2.imread(court_path)
        pts1 = np.float32([[0, 0],[0, -18],[9, -18]])
        pts2 = np.float32([[64, 577],[64, 50],[344, 50]])

        self.M = cv2.getAffineTransform(pts1, pts2)

    # Make output image
    def make_image(self, cx,cy):
        point = self.M @ (np.array([cx,cy,1]).T)
        court_copy = self.court_image.copy()
        court_copy = cv2.circle(court_copy, (int(point[0]),int(point[1])), 10, (0,0,0), -1)

        return court_copy
    
    def make_line(self, line1, line2):
        court_copy = self.court_image.copy()
        for line_ in [line1,line2]:
            p0 = line_(0)
            p1 = line_(50)

            p0 = self.M @ (np.array([p0[0],p0[1],1]).T)
            p1 = self.M @ (np.array([p1[0],p1[1],1]).T)
            image = cv2.line(court_copy, (int(p0[0]),int(p0[1])), (int(p1[0]),int(p1[1])), (0,0,0), 5)

        # plt.imshow(image);plt.show()
        return image





# X = Court()
# plt.imshow(cv2.cvtColor(X.make_image(4.5,-3), cv2.COLOR_BGR2RGB));plt.show()

