import ntpath 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.ticker as plticker


def find_local_peaks(dataFrame,col):
    dataFrame.grd_column=col

    dataFrame['min']=dataFrame.grd_column[(dataFrame.grd_column.shift(1) >= dataFrame.grd_column) & (dataFrame.grd_column.shift(-1) >=dataFrame.grd_column)]
    return dataFrame

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


file_paths=["D:/Research/codes copy/Key Presses and Gradients/With Gradient/With Gradient0613-M.csv",
"D:/Research/codes copy/Key Presses and Gradients/With Gradient/With Gradient1369-M.csv",
"D:/Research/codes copy/Key Presses and Gradients/With Gradient/With Gradient1478-M.csv",
"D:/Research/codes copy/Key Presses and Gradients/With Gradient/With Gradient1479-M.csv",
"D:/Research/codes copy/Key Presses and Gradients/With Gradient/With Gradient1593-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient1789-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient1920-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient2589-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient2650-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient3571-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient3592-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient5476-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient5492-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient6831-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient7153-M.csv",
"D:\Research\codes copy\Key Presses and Gradients\With Gradient\With Gradient9416-M.csv"]

for a_file in file_paths:

    df1=pd.read_csv(a_file)
    
    # data='Gradient'
    df2=find_local_peaks(df1,df1.Gradient)

    fig=plt.figure()
    ax =fig.add_subplot()
    intervals = 30
    loc = plticker.MultipleLocator(base=intervals)
    loc1 = plticker.MultipleLocator(base=15)
    ax.xaxis.set_minor_locator(loc)
    ax.xaxis.set_major_locator(loc1)
    ax.grid(which='minor', axis='x', linestyle='-')
    ax.grid(which='major', axis='x', linestyle='-',color='g')
    ax.scatter(df2['Frame_No'],df2['min'], c='r')
    ax.plot(df2.Frame_No,df2.Gradient)
    ax.set_ylim(-5,10)
    ax.title.set_text('key press without smoothing')
    plt.show()

    # print(df2)
    # print(df)
    
    
    filename=path_leaf(a_file)
    figname=str(filename).split("-")
    print("______________")
    print(figname[0])
    print("______________")
    
    fignamestr=figname[0]
    # print(type(fignamestr))
    fignamestr=fignamestr +".png"
    # print(df1)
    df2=df2.dropna(axis=0)
    # print(df2)
    

    df2 = df2.drop(df2[df2.Y > 630].index)
    df2 = df2.drop(df2[df2.Gradient > 2].index)
    # print(df1)
    y=np.array(df2['Gradient'])
    x=np.array(df2['Frame_No'])
    print(x)
    print(y)

    data=np.vstack((x,y)).T 
    # print(type(data))

    # plt.scatter(data[:,0],data[:,1], label='True Position')
    # plt.show()

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)

    fig,ax=plt.subplots(2,1)
    ax[0].scatter(data[:,0],data[:,1], label='True Position')
    ax[0].title.set_text("Valleys")
    ax[1].scatter(data[:,0],data[:,1], c=kmeans.labels_, cmap='rainbow')
    # plt.scatter(data[:,0],data[:,1], c=kmeans.labels_, cmap='rainbow')
    ax[1].scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    ax[1].title.set_text("Clusters of Valleys")
    plt.show()
    fig.savefig(fignamestr)


    frame_no= np.round(kmeans.cluster_centers_[:,0])
    frame_no=np.sort(frame_no)
    print(frame_no)




    
    df=pd.read_csv(a_file)
    df= df[df['Frame_No'].isin(frame_no)]
    print(df)
    filename=path_leaf(a_file)
    df.to_csv(filename,index=False)
    