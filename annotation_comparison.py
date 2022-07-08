'''
Program that compares data annotations and converts a Napari csv file to and from a 3D Numpy array.
'''

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import KDTree

def toArray(filename):
    """
    Inputs csv file and outputs Numpy array
    """
    df = pd.read_csv(filename,usecols= ['axis-0','axis-1','axis-2'])
    df.columns = ['Z','X','Y']
    df = df[['X','Y','Z']]
    return df.to_numpy()

def toCSV(arr, filename):
    """
    Changes xyz array back to napari csv format
    """
    newArr = []
    for slice in range(len(arr)):
        for coord in arr[slice]:
            newArr.append([float(slice),coord[0],coord[1]])

    df = pd.DataFrame(newArr)
    df = df.reset_index()
    df.columns = ['index','axis-0','axis-1','axis-2']
    
    df.to_csv(filename, sep=',',index=None)

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

def bruteForceCompare(seg1filename, seg2filename, tolerance):
    seg1, seg2 = toArray(seg1filename), toArray(seg2filename)
    # arr1 and arr2 are 3D arrays where arr[n] = nth slice, arr[n][k] = kth coordinate pair
    # arr3 contains overlapping neurons
    # assumption is that imaging starts at slice 0
    slices = int(max(seg1[-1][2], seg2[-1][2])) + 1
    arr1, arr2, overlap = [[] for i in range(slices)], [[] for i in range(slices)], [[] for i in range(slices)]

    for coord in seg1:
        arr1[int(coord[2])].append([coord[0],coord[1]])
    for coord in seg2:
        arr2[int(coord[2])].append([coord[0],coord[1]])
    
    # O(n^3) brute force method
    for t in range(slices):
        for coord1 in arr1[t]:
            closest = tolerance
            for coord2 in arr2[t]:
                # determine which point in arr2 is closest to coord1 in arr1
                if math.dist([coord1[0], coord1[1]], [coord2[0], coord2[1]]) < closest:
                    closest = math.dist([coord1[0], coord1[1]], [coord2[0], coord2[1]])
            if closest != tolerance:
                overlap[t].append([(coord1[0]+coord2[0])*0.5, (coord1[1]+coord2[1])*0.5]) # average
    
    return overlap

KDTree.spatial_distance_matrix

def slidingCompare():
    # sort coordinates by sum
    # sorted(coordinates, key = lambda x: sum(x))
    # sliding window technique?
    pass

def treeCompare():



def partitionCompare(seg1filename, seg2filename, fileDimension, tolerance):
    # place points in arr1 in array, save location of points in the array in another array "arr1loc"
    seg1, seg2 = toArray(seg1filename), toArray(seg2filename)
    # arr1 and arr2 are 3D arrays where arr[n] = nth slice, arr[n][k] = kth coordinate pair
    # arr3 contains overlapping neurons
    # assumption is that imaging starts at slice 0
    slices = int(max(seg1[-1][2], seg2[-1][2])) + 1
    arr1, arr2, overlap = [[] for i in range(slices)], [[] for i in range(slices)], [[] for i in range(slices)]

    # grid partitioned into tolerance x tolerance squares
    grid = [[[[[] for i in range(2)] for col in range(fileDimension // tolerance + 1)] for row in range(fileDimension // tolerance + 1)] for i in range(slices)]
    # need to convert to numpy arrays

    for coord in seg1:
        arr1[int(coord[2])].append([coord[0],coord[1]])
    for coord in seg2:
        arr2[int(coord[2])].append([coord[0],coord[1]])
    
    # O(n^3) brute force method
    for t in range(slices):
        pass
    # place points in arr2 in array
    # go through each point in arr1loc
    # test all the arr2 points in the 9 boxes and see which one is the closest if any
    return arr1

def main():
    # overlap = bruteForceCompare("seg1_points.csv","seg2_points.csv", 2)
    # toCSV(overlap,"overlap_points.csv")
    overlap = partitionCompare("seg1_points.csv","seg2_points.csv", 200, 2)

if __name__ == "__main__":
    main()