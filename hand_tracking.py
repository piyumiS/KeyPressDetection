import cv2
import imutils
import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
print(cv2.__version__)
# "D:\Research\Cropped videos\Video 1.mp4"
# video=cv2.VideoCapture("D:/Research/Cropped videos/Video 2.mp4")
# video=cv2.VideoCapture("C:/Users/Piyumi/Downloads/04..6831.mp4")
# "C:\Users\Piyumi\Videos\VID_20191003_084731.mp4"
video=cv2.VideoCapture("C:/Users/Piyumi/Downloads/4 - Copy.mp4")



# creating mask
blue =103 
green = 199
red = 51
 
color = np.uint8([[[blue, green, red]]])
hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
 
hue = hsv_color[0][0][0]
print  (hue)
low=hue-10
high=hue+10

 
print("Lower bound is :"),
print("[" + str(hue-10) + ", 100, 100]\n")
 
print("Upper bound is :"),
print("[" + str(hue + 10) + ", 255, 255]")


lower_range = np.array([low, 100, 100], dtype=np.uint8) 
upper_range = np.array([high, 255, 255], dtype=np.uint8)

x2d=[]
y2d=[]
listz=[]


# _, first_frame=video.read()
# -----------------------------------------------
while True:
    _, frame = video.read()
    frame_no = int(video.get(0))
    

    listz.append(frame_no)
    # print(frame_no)                                           
    
    if frame is None:
        
        # darray=np.dstack([xvalues,yvalues,zvalues])
        # y=np.gradient(darray,axis=-1)
        # print(y)

        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot(xvalues, yvalues, zvalues, label='3D curve')
        ax.legend()
        number_of_ticks = 20
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$frame no $',rotation = 0)

       
        ax.zaxis.set_ticks(np.arange(min(zvalues),max(zvalues)+(max(zvalues)-min(zvalues))/number_of_ticks, ((max(zvalues)-min(zvalues))/number_of_ticks )))
        plt.show()



        
        fig.savefig("3d.png")
        plt.show()

        print('yvalues')
        print(len(yvalues))
        print('xvalues')
        print(len(xvalues))
        print('zvalues')
        print(len(zvalues))

        xyzdf=pd.DataFrame({'x':xvalues,'y':yvalues,'z':zvalues})
        # print(xyzdf)
        print(xyzdf.shape)
        xyzdf.to_csv('xyz_data',index=False)
        
        y2 = np.gradient(yvalues, 1)
        x2 = np.gradient(xvalues, 1)
        xabs=np.absolute(x2)
        yabs=np.absolute(y2)

        xsq=[x**2 for x in xabs]
        ysq=[y**2 for y in yabs]
        print(xabs)
        print(yabs)

        print(xsq)
        print(ysq)

        
        sq=np.add(xsq,ysq)
        print(sq)
        xysqrt= np.sqrt(sq)
        fig=plt.figure()
        ax =fig.add_subplot()


        ax.plot(zvalues, xysqrt)
        fig.savefig("gradient.png")
        plt.show()
        mydata=pd.DataFrame(xysqrt,zvalues)
        mydata.to_csv('gradient_data')

        


        

        

        


        break
    else:
        frame=imutils.rotate(frame,270)
        cv2.namedWindow('output2',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output2',600,600)
        cv2.imshow('output2', frame)

    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    

    
    
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
        # print(x,y)
        M=cv2.moments(c)
        # print(M)
        center= (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
		# cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # print(center)
    xycoord=list(center)
    x2d.append(xycoord[0])
    y2d.append(xycoord[1])
    
    zvalues=np.array(listz) 
    xvalues=np.array(x2d) 
    yvalues=np.array(y2d) 





    



  
    



    cv2.namedWindow('output1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output1',600,600) 
    cv2.imshow('output1',mask)

    


    key =  cv2.waitKey(10)
    if key == 60:
        break







video.release()
cv2.destroyAllWindows()

