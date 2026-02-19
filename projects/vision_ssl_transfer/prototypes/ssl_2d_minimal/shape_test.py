import matplotlib.pyplot as plt
from samples import *

x, y = generate_positions(128)

center = [0.5, 0.5]
base = 0.3
height = 0.2

sample = triangle(x, y, center, base, height)

plt.imshow(sample)
plt.show()
