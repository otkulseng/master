import storage

from scipy.interpolate import RegularGridInterpolator
import numpy as np


store = storage.load("neel2")


experiment = store.get("85a87b85dd4c487090c990ded2d5d57e")

energy = experiment["condensation_energy"][:]
tolerance = experiment["tol"][: energy.shape[0]]
gap = experiment["order_params"][: energy.shape[0]]



import pandas as pd


df = pd.DataFrame(energy).rename(
    columns={0: "kmode", 1: "t", 2: "mu", 3: "V", 4: "m", 5: "r", 6: "F"}
)

df["tol"] = np.linalg.norm(tolerance, axis=-1)
for i in range(gap.shape[-1]):

    df[f'gap{i}'] = gap[:, i]

df["F"] = df["F"].clip(
    upper=0
)  # All condensation energies that are positive correspond to 'erroneous' physical state
# print(df)
# assert(False)

df = df.groupby(by=["t", "mu", "V", "m", "r"]).mean().drop(columns=["kmode"])
# Now, index is T-mu-V-m-r vs F

# df = df.pivot(index=['t'], columns=['F'])

groups = df.groupby(by=["mu", "V", "m", "r"])
print(len(groups))
# assert(False)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=1, figsize=(10, 10))
for elem in groups:
    X, Y = elem


    y = np.zeros(gap.shape[-1])

    for i in range(y.shape[0]):
        y[i] = np.real(Y[f'gap{i}'])

    print(Y)


    X, Y = elem
    mu, V, m, r = X

    # print(Y.index)
    Y = Y.reset_index()
    # print(Y)

    # T = Y["t"].to_numpy()
    print(y.shape)
    r = int(r)
    # assert(False)
    # y = np.roll(y,  -r)
    x = np.arange(y.size) + 0.5 * r%2

    ax.plot(x, y, label=f'r={r}')

    # name = str(X)

plt.legend()
ax.set_title("Gap")
fig.supxlabel("Temperature[t]")
fig.savefig("img2/" + "all" + ".pdf")
plt.close()
# print(df)

# print(df.columns, df.r)

# [0, 1, 2, 3, 4, 5, 6]
# [r, free_energy]

# .groupby(3).sum().get(4)#.to_numpy()

# T = arr.index.to_numpy()
# vals = arr.to_numpy()


# import matplotlib.pyplot as plt
# plt.plot(T, vals)
# plt.savefig('condenergy.pdf')


# print(df)
