import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from turorials.perlin_noise.perlin_noise import generate_perlin_noise_2d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

SHAPE = 16
RES = 1

x_coord = np.tile(np.array(range(SHAPE)), (SHAPE, 1))
y_coord = np.transpose(x_coord)
z_coord = generate_perlin_noise_2d((SHAPE, SHAPE), (RES, RES))

print(z_coord[1][1])
print(z_coord.shape)

# ax.plot_wireframe(x_coord, y_coord, z_coord, rstride=2, cstride=2)
ax.plot_surface(x_coord, y_coord, z_coord, rstride=1, cstride=1, cmap=cm.coolwarm)

plt.show()