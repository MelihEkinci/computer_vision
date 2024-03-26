import numpy as np
import matplotlib.pyplot as plt

array = np.load('01/IMG_9939.npy')
print('Loaded array of size', array.shape)
print('The pens, from top to bottom, are red, green and blue')

plt.imshow(array)
plt.show()