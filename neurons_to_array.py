'''
Program that converts a csv file from Napari to a 3D Numpy array.
'''

import pandas as pd

filename = 'example_neurons.csv'
df = pd.read_csv(filename,usecols= ['axis-0','axis-1','axis-2'])
df.columns = ['Z','X','Y']
df = df[['X','Y','Z']]
df.to_numpy()

print(df)