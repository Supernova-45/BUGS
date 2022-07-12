'''
Program that compares segmenters' data annotations from a Napari/Fiji CSV file.
'''

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import cKDTree

from matplotlib_venn import venn3

def napari_to_array(filename, resolution):
    """
    Input: Napari CSV file; Output: NumPy array
    """
    df = pd.read_csv(filename,usecols= ['axis-0','axis-1','axis-2'])
    df.columns = ['Z','Y','X']
    df = df[['X','Y','Z']]
    df['Z'] *= 4.8 # is this a universal measurement?
    df['X'] *= resolution
    df['Y'] *= resolution
    return df.to_numpy()

def fiji_to_array(filename, resolution):
    """
    Input: Fiji CSV file, pixels per micrometer; Output: NumPy array
    """
    df = pd.read_csv(filename,usecols= ['X','Y','Slice'])
    df['Slice'] -= 1 # calibrate because fiji slices start at 1; napari starts at 0
    df['Slice'] *= 4.8 # slices to um
    # ***doesn't fiji default to um already?****
    # df['X'] /= resolution 
    # df['Y'] /= resolution 
    return df.to_numpy()

def to_napari_csv(arr, filename):
    """
    Turns 2D XYZ array of points into Napari CSV format
    Input: NumPy array, String (name of output file); Output: CSV file called "filename"
    """
    df = pd.DataFrame(arr)
    df = df.reset_index()
    df.columns = ['index','axis-1','axis-2','axis-0']
    df = df[['index','axis-0','axis-1','axis-2']]

    df.to_csv(filename, sep=',',index=None)

def plot(arr):
    """
    Plots a 3D array of data points in Matplotlib
    Inputs: A 2D array; each entry is [arr, plotColor, labelName] 
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for info in arr:
        ax.scatter3D(info[0][:,0],info[0][:,1],info[0][:,2],s=20,color=info[1],label=info[2])

    fig.canvas.set_window_title('Ok3')
    ax.set_title("Zebrafish brain neurons", fontweight = 'bold')
    ax.set_xlabel('X-axis', fontweight = 'bold')
    ax.set_ylabel('Y-axis', fontweight = 'bold')
    ax.set_zlabel('Z-axis (slice)', fontweight = 'bold')

    plt.legend(loc="upper right")
    plt.show()

def venn_three(tuple,label1,label2,label3,color1,color2,color3,alpha):
    """
    Input: tuple = (a,b,ab,c,ac,bc,abc), three labels and colors are strings
    Output: Weighted venn diagram with three circles
    """
    venn3(subsets = tuple, set_labels = (label1,label2,label3), set_colors = (color1,color2,color3), alpha = alpha)
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
    return round(matched*100,4), round(mismatched1*100,4), round(mismatched2*100,4)

def two_segs(arr1, arr2, radius, nameOne, nameTwo):
    """
    Returns information about the points clicked by two segmenters
    Input: Numpy arrays from two segmenters, tolerance threshold, name of two segmenters (String)
    Output: String
    """
    matched12, mismatched12, mismatched21 = percent_matched(arr1, arr2, radius)
    msg = f"Seg {nameOne} and seg {nameTwo} overlapped by {matched12} percent, with {mismatched12} percent mismatched by seg {nameOne} and {mismatched21} percent mismatched by seg {nameTwo}.\n"
    return msg

def three_segs(arr1, arr2, arr3, radius):
    """
    Returns information about combinations of three segmenters
    Input: Numpy arrays from two segmenters, tolerance threshold, name of two segmenters (String)
    Output: String
    """
    msg = two_segs(arr1, arr2, radius, "1", "2") + two_segs(arr1, arr3, radius, "1", "3") + two_segs(arr2, arr3, radius, "2", "3")
    return msg

def venn_three_sizes(arr1, arr2, arr3, radius):
    x, y = nearest_pairs(arr1, arr2, radius)
    ab = overlap_size(x)
    overlap = np.empty((ab,3))
    for i in range(x.len()):
        if a != -1:
            np[i] = a[i]
    x, y = nearest_pairs(overlap, arr3, radius)
    abc = overlap_size(x)

    x, y = nearest_pairs(arr1, arr3, radius)
    ac = overlap_size(x)
    x, y = nearest_pairs(arr2, arr3, radius)
    bc = overlap_size(x)

    # By principle of inclusion-exclusion
    a = arr1.len()-ab-ac+abc
    b = arr1.len()-ab-bc+abc
    c = arr1.len()-ac-bc+abc

    return (a,b,ab,c,ac,bc,abc)

def many_segs(*args):
    pass

def main():
    # seg1 = napari_to_array("data/seg1_points.csv")
    # seg2 = napari_to_array("data/seg2_points.csv")
    # print(two_segs(seg1, seg2, 2, "1", "2"))

    suhan = fiji_to_array("data/suhan_7_9_2022.csv",1.7)
    lindsey = np.concatenate((napari_to_array("data/lindsey_sn_7_9_2022.csv"), napari_to_array("data/lindsey_mn_7_9_2022.csv")), axis=0)
    alex = napari_to_array("data/alex_7_9_2022.csv")

    # plot([[suhan, 'red', 'suhan'],[lindsey, 'green', 'lindsey'],[alex, 'blue', 'alex']])

    # print(three_segs(alex, lindsey, suhan, 4))

    # plotThree(suhan, 'red', 'suhan',lindsey, 'green', 'lindsey',alex, 'blue', 'alex')
    # fiji = fiji_to_array("/Users/alexandrakim/Desktop/BUGS2022/fiji_two_points.csv")
    # napari = napari_to_array("/Users/alexandrakim/Desktop/BUGS2022/napari_two_points.csv")

    # to_napari_csv(fiji, "data/output_points.csv")
    # plot([[fiji, 'red', 'fiji'],[napari, 'green','napari']])

if __name__ == "__main__":
    main()