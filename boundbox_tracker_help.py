import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_cx_cy(box):
    # pass the coordinates of the box
    # return the coordinates of the centroid of the bounding box
    # later this coordinates will be used to draw the 3-D graph
    x,y,w,h=[int(c) for c in box ]
    x2=x+w
    y2=y+h
    cx= (x+x2)/2
    cy=(y+y2)/2
    return cx,cy
# following lines can be used to understand the function
tupx=(2,5,1,1)
x,y=get_cx_cy(tupx)
print(int(x),int(y))

def draw_xyz3D(x,y,z):
    
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(x, y, z, label='3D curve')
    ax.legend()
    number_of_ticks = 20
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$frame no $',rotation = 0)

    
    ax.zaxis.set_ticks(np.arange(min(z),max(z)+(max(z)-min(z))/number_of_ticks, ((max(z)-min(z))/number_of_ticks )))
    plt.show()    
    fig.savefig("3d.png")
    plt.show()

def draw_gradient_graph(x,y):
    y2 = np.gradient(y, 1)
    x2 = np.gradient(x, 1)
    xabs=np.absolute(x2)
    yabs=np.absolute(y2)

    xsq=[x**2 for x in xabs]
    ysq=[y**2 for y in yabs]
    # print(xabs)
    # print(yabs)

    # print(xsq)
    # print(ysq)

    
    sq=np.add(xsq,ysq)
    # print(sq)
    xysqrt= np.sqrt(sq)
    fig=plt.figure()
    ax =fig.add_subplot()


    ax.plot(zvalues, xysqrt)
    fig.savefig("gradient.png")
    plt.show()
    mydata=pd.DataFrame(xysqrt,zvalues)
    mydata.to_csv('gradient_data')
    return mydata