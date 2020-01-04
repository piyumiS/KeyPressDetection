import pandas as pd 
import numpy as np 

def make_arrays(xlist,ylist,zlist):
    xarray=np.array(xlist) 
    yarray=np.array(ylist) 
    zarray=np.array(zlist)
    return xarray,yarray,zarray

def save_xyz_to_df(x,y,z,file_name):
    # save x,y coordinates of marker with respective frame no.
    #  in a csv file by creating a pandas dataframe. 
    # for each run/each video file pass a different filename. otherwise 
    # file will get replaced in each run.
    
    xyzdf=pd.DataFrame({'X':x,'Y':y,'Frame No':z})
    xyzdf.to_csv(file_name,index=False)
    return xyzdf


def cal_gradient(x_array,y_array):

    # get the gradients of both x and y directions.
    # here we don't consider where it is negative or positive direction on each axis
    # therefore get the absolute value of the gradient and 
    # get the net change in both direction for each frame

    # calculate the gradient
    gradient_of_x=np.gradient(x_array,0.1)
    gradient_of_y=np.gradient(y_array,0.1)
 
    sq_grd=np.add([x**2 for x in gradient_of_x],[y**2 for y in gradient_of_y])

    sqrt_grd=np.sqrt(sq_grd)

    return  sqrt_grd
# x=[1,2,3]
# y=[4,5,6]
# xx=np.array(x)
# yy=np.array(y)
# grad_check=cal_gradient(xx,yy)
# print (grad_check)