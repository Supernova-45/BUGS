'''
Program that compares segmenters' data annotations from a Napari/Fiji CSV file.
'''

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import KDTree

def napariToArray(filename):
    """
    Input: Napari CSV file; Output: NumPy array
    """
    df = pd.read_csv(filename,usecols= ['axis-0','axis-1','axis-2'])
    df.columns = ['Z','X','Y']
    df = df[['X','Y','Z']]
    return df.to_numpy()

def fijiToArray(filename):
    """
    Input: Fiji CSV file; Output: NumPy array
    """
    pass

def toCSV(arr, filename):
    """
    Changes 3D XYZ array of points into Napari CSV format
    Input: NumPy array, name of output file; Output: CSV file called "filename"
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
    """
    Plots a 3D array of datapoints in Matplotlib
    Inputs: 3D array, color of the points, label to add to the legend
    """
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

def getOverlap(arr1, arr2):
    """
    Computes the points selected by two segmenters by calculating nearest neighbors within a certain tolerance
    Inputs: two 3D arrays (x, y, slice)
    Output: a 1D array with length of arr1.len()
    Code adapted from X___
    """
    pass

def overlapSize(arr): 
    """
    Finds the number of non -1 points from a 1D array
    Input: 1D array from getOverlap(); Output: int
    """

def percentMatched(arr1, arr2):
    """
    Computes the percent of points clicked by 2 segmenters
    Input: two 3D NumPy arrays; Output: Float between 0 and 1
    """
    pass

def percentMisMatched(arr1, arr2):
    """
    Computes the percent of points which only one segmenter clicked
    Input: Output: [float, float], or [seg1 % mismatched, seg2 % mismatched]
    """
    pass

def manySegmenters(*args):
    """
    Returns an array of points clicked by 3+ segmenters
    """
    pass

def tiffToArray(filename):
    """
    Utilizes imageManipulation class
    Input: TIFF file; Output: NumPy array
    """
    pass

def shrinkGrid(arr, n, x, y, z):
    """
    Input: 3D NumPy array from TIFF file; Output: n x n portion of the grid centered at (x,y,z)
    """
    pass

def neuronBrightness(arr):
    """
    Determine the local brightness of a neuron 
    Input: Numpy array representing small portion of TIFF file; Output: Float 
    """
    pass

def segBrightness(arr, filename):
    """
    Returns the brightness of neurons the segmenter clicked
    Input: 3D NumPy array (points clicked), TIFF file; Output: Float (average brightness)
    """
    pass

def bruteForceCompare(seg1filename, seg2filename, tolerance):
    seg1, seg2 = napariToArray(seg1filename), napariToArray(seg2filename)
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

def main():
    # overlap = bruteForceCompare("seg1_points.csv","seg2_points.csv", 2)
    # toCSV(overlap,"overlap_points.csv")
    overlap = partitionCompare("seg1_points.csv","seg2_points.csv", 200, 2)

if __name__ == "__main__":
    main()