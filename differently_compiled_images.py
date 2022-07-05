from PIL import Image
import numpy as np

im = Image.open('example_movie.tif')
im.show()

arr = np.array(im)
print(arr.shape)