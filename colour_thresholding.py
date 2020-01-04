import sys
import cv2
import imutils
import pandas as pd 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splrep, splev
from scipy.signal import medfilt
from colour_thresh_helpers import *
from get_key_press import *
# check opencv version
print(cv2.__version__)

# insert the RGB value of the color of the object 
# that is going to be tracked
# you can get the RGB values by using video/image editor
# ex: I saved a moment from the video using movies&TV and then open the saved image in paint 3D to get the RGB values of the TRACKER pasted on the hand
# note that opencv reads BGR not RGB
blue=56
green= 94
red=73
colour=np.uint8([[[blue,green,red]]])

# convert colour to HSV values to create mask
hsv_col=cv2.cvtColor(colour,cv2.COLOR_BGR2HSV)
hue=hsv_col[0][0][0]
print(hue)

# define lower bound and upper bound of the color to track
lower=hue-20
upper=hue+20


print("Lower bound-->")
print("["+ str(lower) + ",100,100]")

print("Upper bound-->")
print("["+ str(upper) + ",255,255]")

lower_range = np.array([lower, 100, 100], dtype=np.uint8) 
upper_range = np.array([upper, 255, 255], dtype=np.uint8)

# Read video
# Copy the video path with the forwardslash
my_video_file="C:/Users/Piyumi/Videos/New Data/1478-M.mp4"
video=cv2.VideoCapture(my_video_file)
v_name="1478_M"
file_name=v_name +".csv"
file_name2= v_name + " With Gradient "
figure_name= v_name+".png"
figure_name3d= v_name+"3d.png"

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()



# initialize 03 lists to store X,Y coordinates of center of the marker and corresponding frame number.
xlist=[]
ylist=[]
z_frame_no=[]

# track the marker throughout each video frame
while True:
    ret,frame=video.read()
    # Get the frame number and append it to z_frame_no
    frame_no=int(video.get(1))
    

    frame=cv2.rotate(frame,cv2.ROTATE_180)
    if frame is not None:
        print("pass")
        # cv2.namedWindow('output2',cv2.WINDOW_NORMAL)
        # # cv2.resizeWindow('output2',600,600)
        # cv2.imshow('output2', frame)

        

       
        # apply mask to find marker
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)        
        mask = cv2.inRange(hsv, lower_range, upper_range)
        cv2.namedWindow('output1',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output1',600,600) 
        cv2.imshow('output1',mask)
        
        # find contours of the marker
        contours =cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        cnts=imutils.grab_contours(contours)
        center=None

        if len(cnts)>0:
            # print(type(cnts))
            # print(len(cnts))
            # print(cnts)
            c=max(cnts,key=cv2.contourArea)
            ((x,y),radius)=cv2.minEnclosingCircle(c)
            # print("radius")
            # print(radius)
            # print(x,y)
            M=cv2.moments(c)
            # print(M)
            if M['m00'] > 0:
                center= (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # print("center")
                # print(type(center))
                # print(x,y)
                # draw contours on the frame
                cv2.circle(frame, (int(center[0]), int(center[1])), int(radius),(0, 255, 255), 2)
                # if the marker is detected then
                xlist.append(center[0]) 
                ylist.append(center[1])
                z_frame_no.append(frame_no)

                info=[("Frame No",frame_no),("X",center[0]),("Y",center[1])]
                for (i,(k,v))in enumerate(info):
                    text="{}:{}".format(k,v)
                    cv2.putText(frame,text,(800,(i*25)+500),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
        
        cv2.namedWindow('output2',cv2.WINDOW_NORMAL)
        cv2.imshow('output2', frame)
      
    




    key=cv2.waitKey(1)& 0xFF
    if key==ord("q") or frame is None:
        break
 
video.release()
cv2.destroyAllWindows()


print("diagrams goes here")
# make arrays for x,y,z to plot graphs
X,Y,Z=make_arrays(xlist,ylist,z_frame_no)

# save x,y,frame no to csv
df=save_xyz_to_df(X,Y,Z,file_name)
# print(df)

# calculate gradient
gradiet_vals=cal_gradient(X,Y)
# print(gradiet_vals)

# append gradient to the xyz dataframe and save another csv
df["Gradient"]=gradiet_vals
# print(df)
df.to_csv(file_name2,index=False)

# get the actual key presses using colour thresholding for the LED
# get_press function returs a pandas df containing frame on of where actual key press happens
press_df,press_array=get_press(my_video_file)
result_df=df.merge(press_df, left_on="Frame No",right_on="Key_Press")
result_df.to_csv(v_name+"with actual key press.csv",index=False)


frm=np.asarray(result_df[["Frame No"]])
grd=np.asarray(result_df[["Gradient"]])
x2=np.asarray(result_df[["X"]])
y2=np.asarray(result_df[["Y"]])

print(result_df)






# draw 3D graph
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(X,Y,Z, label='3D curve')
ax.legend()
number_of_ticks = 20
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$frame no $',rotation = 0)
ax.zaxis.set_ticks(np.arange(min(Z),max(Z)+(max(Z)-min(Z))/number_of_ticks, ((max(Z)-min(Z))/number_of_ticks )))
plt.show()
fig.savefig(figure_name3d)
plt.show()


# draw gradient graph
Z_new=np.linspace(Z.min(),Z.max(),500)
bspl = splrep(Z,gradiet_vals,s=5)
bspl_y = splev(Z_new,bspl)


smooth_df=pd.DataFrame({'Z':Z_new,'data':bspl_y})

smooth_df.to_csv("smoothen_y.csv",index=False)

# smooth_df['min'] = smooth_df.Y_Smooth[(smooth_df.Y_Smooth.shift(1) > smooth_df.Y_Smooth) & (smooth_df.Y_Smooth.shift(-1) > smooth_df.Y_Smooth)]
# plt.scatter(smooth_df.index, df['min'], c='r')
# smooth_df.Y_Smooth.plot()

# filt_grad=medfilt(gradiet_vals,)

# fig=plt.figure()
fig,ax=plt.subplots(2,1)
ax[0].plot(X,Y)
ax[0].scatter(x2,y2,c="#FF5733")

ax[1].set_ylim(-1,30)

ax[1].plot(Z,gradiet_vals)
ax[1].plot(Z_new,bspl_y,c="#FFFF00")
ax[1].scatter(frm,grd,c="#FF6133")

fig.savefig(figure_name)
plt.show()

 


