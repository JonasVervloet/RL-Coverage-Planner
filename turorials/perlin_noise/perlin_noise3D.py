import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from turorials.perlin_noise.perlin_noise import generate_perlin_noise_2d

SHAPE = 256
RES = 4

x_coord = np.tile(np.array(range(SHAPE)), (SHAPE, 1))
y_coord = np.transpose(x_coord)
z_coord = generate_perlin_noise_2d((SHAPE, SHAPE), (RES, RES))

plt.imshow(z_coord, cmap='gray')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.plot_wireframe(x_coord, y_coord, z_coord, rstride=2, cstride=2)
ax.plot_surface(x_coord, y_coord, z_coord, rstride=16, cstride=16, cmap=cm.coolwarm)

plt.show()