'''
Program that compares segmenters' data annotations from a Napari/Fiji CSV file.
'''

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from scipy.spatial import cKDTree

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
    df = pd.read_csv(filename,usecols= ['X','Y','Slice'])
    return df.to_numpy()

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
    return len(arr) - np.count_nonzero(arr == -1)

def percent_matched(arr1, arr2, radius):
    """
    Computes the percent of points clicked by 2 segmenters and the percent of points which only one segmenter clicked
    Input: two 3D NumPy arrays; Output: Float between 0 and 100
    """
    closestOne, closestTwo = nearest_pairs(arr1, arr2, radius)
    matched = overlap_size(closestOne) / (len(arr1) + len(arr2))
    mismatched1 = (len(arr1) - overlap_size(closestOne)) / (len(arr1) + len(arr2))
    mismatched2 = (len(arr2) - overlap_size(closestOne)) / (len(arr1) + len(arr2))
    return matched*100, mismatched1*100, mismatched2*100

def percent_mismatched(arr1, arr2, radius):
    """
    Computes the percent of points which only one segmenter clicked
    Input: Output: float, float -- seg1 % mismatched, seg2 % mismatched
    """
    pass

def two_segs(arr1, arr2, radius, one, two):
    matched12, mismatched12, mismatched21 = percent_matched(arr1, arr2, radius)
    msg = f"Seg {one} and seg {two} overlapped by {matched12} percent, with {mismatched12} percent mismatched by seg {one} and {mismatched21} percent mismatched by seg {two}.\n"
    return msg

def three_segs(arr1, arr2, arr3, radius):
    """
    Returns percent of points clicked by combinations of the 3 segmenters
    Input: 3 segmenters' arrays and tolerance; Output: 2D array [1&2&3, [1, 1&2, 1&3], [2, 2&1, 2&3], [3, 3&1, 3&2]]
    """
    msg = two_segs(arr1, arr2, radius, "1", "2") + two_segs(arr1, arr3, radius, "1", "3") + two_segs(arr2, arr3, radius, "2", "3")
    return msg

def manySegs(*args):
    pass

def main():
    # seg1 = napari_to_array("seg1_points.csv")
    # seg2 = napari_to_array("seg2_points.csv")
    # print(two_segs(seg1, seg2, 2, "1", "2"))

    suhan = fiji_to_array("/Users/alexandrakim/Desktop/BUGS2022/suhan_7_9_2022.csv")
    lindsey = np.concatenate(napari_to_array("/Users/alexandrakim/Desktop/BUGS2022/lindsey_sn_7_9_2022.csv"),napari_to_array("/Users/alexandrakim/Desktop/BUGS2022/lindsey_mn_7_9_2022.csv"), axis=0)
    alex = napari_to_array("/Users/alexandrakim/Desktop/BUGS2022/7_5_2022_left_forebrain_1P.csv")
    
if __name__ == "__main__":
    main()