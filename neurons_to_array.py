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

def plot(arr):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.scatter3D(arr[:,0],arr[:,1],arr[:,2],color='blue')

    plt.title("Zebrafish brain neurons")
    ax.set_xlabel('X-axis', fontweight = 'bold')
    ax.set_ylabel('Y-axis', fontweight = 'bold')
    ax.set_zlabel('Z-axis', fontweight = 'bold')

    plt.show()

def main():
    arr1 = toArray("example_neurons.csv")
    plot(arr1)

if __name__ == "__main__":
    main()