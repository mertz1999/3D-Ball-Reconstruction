# 3D Ball Reconstruction
In this repository we use outputs of TrackNet models for making 3D trajectory. 

Tabel of contents:
- Make Pairs of 3D point and 2D point


## Make Pairs
First step to using this repository you need to make at least 6 pairs of 3D/2D point that belongs to your video cameras.

so use this block of code to make that.

name variable is used to save a .npy file with that name. 

data is used to defining pairs. each row contain 5 element that fist 3 element is X,Y,Z point in 3D volleyball court and last 2 element is x,y point in image format.
```python
name = '6_p1'
data = [
    [0,-6,0, 155, 480],
    [0,-12,0, 227, 333],
    [9,-12,0, 1076, 331],
    [9,-6,0, 1139, 479],
    [0,-9,2.43, 183, 155],
    [9,-9,2.43, 1111, 153],
]
data = np.array(data)
np.save('./data/'+name+'.npy', data)
```
you can see ***make_data.py*** file for usage this code.

3D important point is like this:

<!-- ![App Screenshot](./inc/3d_court.jpg) -->

<p align="center">
<img src="./inc/3d_court.jpg" style="width:60%;"/>
</p>