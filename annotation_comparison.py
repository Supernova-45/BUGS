'''
Program that compares segmenters' data annotations from a Napari/Fiji CSV file.
'''

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import cKDTree

def napari_to_array(filename):
    """
    Input: Napari CSV file; Output: NumPy array
    """
    df = pd.read_csv(filename,usecols= ['axis-0','axis-1','axis-2'])
    df.columns = ['Z','X','Y']
    df = df[['X','Y','Z']]
    return df.to_numpy()

def fiji_to_array(filename):
    """
    Input: Fiji CSV file; Output: NumPy array
    """
    pass

def to_csv(arr, filename):
    """
    Turns 3D XYZ array of points into Napari CSV format
    Input: NumPy array, String (name of output file); Output: CSV file called "filename"
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
    Plots a 3D array of data points in Matplotlib
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
    
def nearest_pairs(v1, v2, radius):
    """
    Adopted from synspy: https://github.com/informatics-isi-edu/synspy.git
    Find nearest k-dimensional point pairs between v1 and v2 and return via output arrays.

       Inputs:
         v1: array with first pointcloud with shape (n, k)
         kdt1: must be cKDTree(v1) for correct function
         v2: array with second pointcloud with shape (m, k)
         radius: maximum euclidean distance between points in a pair

       Use greedy algorithm to assign nearest neighbors without
       duplication of any point in more than one pair.

       Outputs:
         out1: for each point in kdt1, gives index of paired point from v2 or -1
         iv1_for_v2: out2: for each point in v2, gives index of paired point from v1 or -1

    """
    depth = min(max(v1.shape[0], v2.shape[0]), 100)

    out1 = np.full((v1.shape[0], 1), -1)
    out2 = np.full((v2.shape[0], 1), -1)

    kdt1 = cKDTree(v1)
    dx, pairs = kdt1.query(v2, depth, distance_upper_bound=radius)
    for d in range(depth):
        for idx2 in np.argsort(dx[:, d]):
            if dx[idx2, d] < radius:
                if out2[idx2] == -1 and out1[pairs[idx2, d]] == -1:
                    out2[idx2] = pairs[idx2, d]
                    out1[pairs[idx2, d]] = idx2
    iv1_for_v2 = out2
    return out1, iv1_for_v2

def overlap_size(arr): 
    """
    Finds the number of non -1 points from a 1D array
    Input: 1D array from getOverlap(); Output: int
    """
    pass

def percent_matched(arr1, arr2):
    """
    Computes the percent of points clicked by 2 segmenters
    Input: two 3D NumPy arrays; Output: Float between 0 and 1
    """
    pass

def percent_mismatched(arr1, arr2):
    """
    Computes the percent of points which only one segmenter clicked
    Input: Output: [float, float], or [seg1 % mismatched, seg2 % mismatched]
    """
    pass

def three_segs(arr1, arr2, arr3):
    """
    Returns percent of points clicked by combinations of the 3 segmenters
    Input: 3 segmenters' arrays; Output: 2D array [1&2&3, [1, 1&2, 1&3], [2, 2&1, 2&3], [3, 3&1, 3&2]]
    """
    pass

def manySegs(*args):
    pass

def tiffToArray(filename):
    """
    Utilizes imageManipulation class
    Input: TIFF file; Output: NumPy array
    """
    pass

def shrinkGrid(arr, n, x, y, z):
    """
    Input: 3D NumPy array from TIFF file; Output: n x n array -- portion of the grid centered at (x,y,z)
    """
    pass

def neuronBrightness(arr):
    """
    Determine the local brightness of a neuron by averaging pixels around it
    Input: Numpy array from shrinkGrid(); Output: Float 
    """
    pass

def segBrightness(arr, filename):
    """
    Returns the brightness of neurons the segmenter clicked
    Input: 3D NumPy array (points clicked), TIFF file; Output: Float (average brightness)
    """
    pass

def main():
    pass

if __name__ == "__main__":
    main()