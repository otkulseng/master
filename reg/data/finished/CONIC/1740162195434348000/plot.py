import zarr
import matplotlib.pyplot as plt
z = zarr.open('result.zarr', mode='r')

ar = z[:]

plt.plot(ar)
plt.savefig('temp.pdf')