'''
Program that converts a csv file from Napari to a 3D Numpy array.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def toArray(filename):
    df = pd.read_csv(filename,usecols= ['axis-0','axis-1','axis-2'])
    df.columns = ['Z','X','Y']
    df = df[['X','Y','Z']]
    df = df.to_numpy()
    return df

def plot(arr, plotColor, labelName):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(arr[:,0],arr[:,1],arr[:,2],s=20,label=labelName,color=plotColor)

    fig.canvas.set_window_title('Ok3')
    ax.set_title("Zebrafish brain neurons", fontweight = 'bold')
    ax.set_xlabel('X-axis', fontweight = 'bold')
    ax.set_ylabel('Y-axis', fontweight = 'bold')
    ax.set_zlabel('Z-axis (slice)', fontweight = 'bold')

    plt.legend(loc="upper right")
    plt.show()

def compare(seg1filename, seg2filename):
    seg1, seg2 = toArray(seg1filename), toArray(seg2filename)
    arr1 = [[] for i in range(int(min(seg1[0][2],seg2[0][2])), int(max(seg1[-1][2], seg2[-1][2]))+1)]
    arr2 = [[] for i in range(int(min(seg1[0][2],seg2[0][2])), int(max(seg1[-1][2], seg2[-1][2]))+1)]
    for coord in seg1:
        print(coord[2])
        arr1[int(coord[2])].append([coord[0],coord[1]])
    for coord in seg2:
        arr2[int(coord[2])].append([coord[0],coord[1]])

def main():
    arr1 = toArray("seg1_points.csv")
    print(arr1)
    print(compare("seg1_points.csv","seg2_points.csv"))

if __name__ == "__main__":
    main()