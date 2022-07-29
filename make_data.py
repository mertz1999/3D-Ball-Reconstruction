from turtle import clear
import numpy as np 

# File name 
name = 'game_8'

# Data in this format: X,Y,Z x,y
data = [
    [0,-6,0,100,581],
    [0,-12,0,311,494],
    [9,-12,0,964,485],
    [9,-6,0,1180,562],
    [0,-9,2.24,223,320],
    [9,-9,2.24,1044,308],
]


data = np.array(data)

np.save('./data/'+name+'.npy', data)