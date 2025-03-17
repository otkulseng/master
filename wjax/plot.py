import storage

from scipy.interpolate import RegularGridInterpolator
import numpy as np


store = storage.load("neel")


experiment = store.get("19923205a9a64f5b99eee307501ba818")
energy = experiment["condensation_energy"][:]
tolerance = experiment["tol"][: energy.shape[0]]
gap = experiment["order_params"][: energy.shape[0]]

tolerance = np.linalg.norm(tolerance, axis=-1)
gap = np.real(np.mean(gap, axis=-1))
# tolerance = np.mean(tolerance, axis=-1)

import pandas as pd

df = pd.DataFrame(energy).rename(
    columns={0: "kmode", 1: "t", 2: "mu", 3: "V", 4: "m", 5: "r", 6: "F"}
)

df["tol"] = tolerance
df['gap'] = gap

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

for elem in groups:
    fig, (ax, axGap, axTol) = plt.subplots(ncols=3, figsize=(15, 5))


    ax.set_title("Condensation energy")
    axGap.set_title("Mean gap")
    axTol.set_title("Norm")

    fig.supxlabel("Temperature[t]")

    X, Y = elem
    mu, V, m, r = X

    # print(Y.index)
    Y = Y.reset_index()
    # print(Y)
    # assert(False)
    F = Y["F"].to_numpy()
    T = Y["t"].to_numpy()

    axTol.scatter(T, np.log10(Y["tol"].to_numpy()))
    axGap.scatter(T, Y["gap"].to_numpy())
    # print(Y)
    # assert(False)

    meanval = np.mean(np.abs(F))

    mask = np.abs(F) < 10 * meanval
    ax.scatter(T[mask], F[mask])

    name = str(X)
    fig.suptitle(f'mu={round(mu, 2)} V={round(V, 2)} m={round(m, 2)} r={r}')
    fig.savefig("img/" + name + ".pdf")
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
