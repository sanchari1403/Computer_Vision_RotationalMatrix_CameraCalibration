###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def getIntrinsicParameters(world_coords,img_points):

    img_points = np.array(img_points).flatten().reshape(36,2)
    a = []
    b = []
    for item in world_coords:
        item_li1 = list(item)
        item_li2 = []
        item_li1.extend([1,0,0,0,0])
        a.append(item_li1)
        item_li2.extend([0,0,0,0])
        item_li2.extend(list(item))
        item_li2.append(1)
        b.append(item_li2)
    arr1 = [i for j in zip(a,b) for i in j]
    li = []
    for item in arr1:
        li.append(np.array(item))
    arr1 = np.array(li,dtype=object)
    li_coords = []
    for i in range(0,36):
        ip = img_points[i]
        w = world_coords[i]
        arr2 = np.array([-ip[0]*w[0], -ip[0]*w[1],-ip[0]*w[2],-ip[0]])
        arr3 = np.array([-ip[1]*w[0], -ip[1]*w[1],-ip[1]*w[2],-ip[1]])
        li_coords.extend([arr2,arr3])
    li_coords = np.array(li_coords).flatten().reshape(72,4)
    arr = np.concatenate((arr1,li_coords),axis=1).astype(float)
    VT = np.linalg.svd(arr)[-1]
    m = VT[-1].reshape((3,4))
    lambda_val = 1 / np.sqrt(np.square(m[2][0]) + np.square(m[2][1]) + np.square(m[2][2]))
    m = lambda_val * m
    m1 = np.array([m[0][0],m[0][1],m[0][2]]).T
    m2 = np.array([m[1][0],m[1][1],m[1][2]]).T
    m3 = np.array([m[2][0],m[2][1],m[2][2]]).T
    ox = np.dot(m1.T,m3)
    oy = np.dot(m2.T,m3)
    fx = np.sqrt((np.dot(m1.T,m1)-ox**2))
    fy = np.sqrt((np.dot(m2.T,m2)-oy**2))
    intrinsic_param = [fx, fy, ox, oy]
    return intrinsic_param

def calibrate(imgname):
    #...... 
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = imread(imgname)
    gray_image = cvtColor(img,COLOR_BGR2GRAY)   
    retval,img_points = findChessboardCorners(gray_image,(4,9))
    corners_grayimg = cornerSubPix(gray_image,img_points, (6,6), (-1,-1), criteria) 
    drawChessboardCorners(img, (4,9), corners_grayimg, retval)
    world_coords = np.array([
        [40,0,40],[40,0,30],[40,0,20],[40,0,10],
        [30,0,40],[30,0,30],[30,0,20],[30,0,10],
        [20,0,40],[20,0,30],[20,0,20],[20,0,10],
        [10,0,40],[10,0,30],[10,0,20],[10,0,10],
        [0,0,40],[0,0,30],[0,0,20],[0,0,10],
        [0,10,40],[0,10,30],[0,10,20],[0,10,10],
        [0,20,40],[0,20,30],[0,20,20],[0,20,10],
        [0,30,40],[0,30,30],[0,30,20],[0,30,10],
        [0,40,40],[0,40,30],[0,40,20],[0,40,10]
    ]) 
    # world_coords_mod = np.array([
    #     [32,0,40],[24,0,30],[40,0,10],[40,0,10],
    #     [30,0,40],[30,0,30],[40,0,10],[30,0,10],
    #     [20,0,40],[15,0,30],[20,0,20],[20,0,10],
    #     [10,0,40],[24,0,30],[16,0,20],[10,0,10],
    #     [0,0,40],[0,0,30],[0,0,13],[0,0,10],
    #     [0,10,16],[0,10,30],[0,10,20],[0,10,10],
    #     [0,20,40],[0,43,30],[0,15,20],[0,20,10],
    #     [0,30,40],[0,30,30],[0,26,20],[0,30,10],
    #     [0,40,32],[0,40,30],[0,40,20],[0,40,10]
    # ])
    intrinsic_params_org = getIntrinsicParameters(world_coords,img_points)
    # intrinsic_params_mod = getIntrinsicParameters(world_coords_mod,img_points)
    is_constant = True
    return intrinsic_params_org, is_constant
    

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)