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
        print(point)
        court_copy = self.court_image.copy()
        court_copy = cv2.circle(court_copy, (int(point[0]),int(point[1])), 10, (0,0,0), -1)

        return court_copy





# X = Court()
# plt.imshow(cv2.cvtColor(X.make_image(30,30), cv2.COLOR_BGR2RGB));plt.show()

