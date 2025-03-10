import yaml
import numpy as np
import matplotlib.pyplot as plt



with open('finished.yaml', 'r') as file:
    data: dict = yaml.load(file, Loader=yaml.UnsafeLoader)

    results = [None] * len(data)
    for key, val in data.items():
        # print(val)
        x, y = np.array(val).T

        results[int(key)] = (x, y)
        # print(key, int(key))

# Free energy vs temperature
plt.figure()
for idx, (x, y) in enumerate(results):
    plt.plot(x, y, label=f'{idx}')

plt.legend()
plt.savefig('FvT.pdf')

# Heatmap

def extend(x, y, temps, default):
    # x should match temps, but if not, add default value
    res_y = []
    for t in temps:
        if t not in x:
            res_y.append(default)
            continue

        idx = np.argwhere(x == t)[0, 0]
        res_y.append(y[idx])
    return res_y




temps = set()
for T, _ in results:
    temps = temps.union(T)
temps = list(temps)
temps.sort()

res = np.zeros((len(results), len(temps)))
DEFAULT = 100
for i, (x, y) in enumerate(results):
    res[i] = extend(x, y, temps, default=DEFAULT)


x = np.arange(len(results))
y = temps

xx, yy = np.meshgrid(x, y, indexing='ij')

plt.figure()
mesh = plt.pcolormesh(xx, yy, res, shading='auto', cmap='viridis')
plt.colorbar(mesh, label='Intensity')
plt.savefig('error.pdf')

for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        if not res[i, j] == DEFAULT:
            continue

        vals = []
        if i > 0:
            vals.append(res[i-1, j])
        if j > 0:
            vals.append(res[i, j-1])
        if i+1 < res.shape[0]:
            vals.append(res[i+1, j])
        if j+1 < res.shape[1]:
            vals.append(res[i, j+1])

        val = np.mean(vals[vals != DEFAULT])
        res[i, j] = val

        # print(i, j)

# print(xx.shape, yy.shape, res.shape)
# assert(False)
plt.figure()
mesh = plt.pcolormesh(xx, yy, res, shading='auto', cmap='viridis')
plt.colorbar(mesh, label='Intensity')
plt.savefig('smooth.pdf')

# Label axes and add a title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Colormesh Plot Example')