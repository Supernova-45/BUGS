import pandas as pd

df = pd.read_csv('example_neurons.csv',usecols= ['axis-0','axis-1','axis-2'])

df.to_numpy()

print(df)