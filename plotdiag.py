import matplotlib.pyplot as plt
import storage

import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from scipy.interpolate import NearestNDInterpolator, CloughTocher2DInterpolator, CubicSpline, griddata, LinearNDInterpolator
import numpy as np
from matplotlib import colors

# store = storage.load("main_test")
# experiment = store.get("d77dab54d19d441da3f54bce6f91d37f")
store = storage.load("testdiag30")
experiment = store.get("4d033f16b74e42f29df3137e1a937294")


energy = experiment["condensation_energy"][:]
tolerance = experiment["tol"][: energy.shape[0]]
gap = experiment["order_params"][: energy.shape[0]]

gap = np.real(np.mean(gap, axis=-1))
# print(gap.shape, np.max(gap), np.min(gap))
# assert(False)
# tolerance = np.linalg.norm(tolerance, axis=-1)


# gap = experiment["order_params"][: energy.shape[0]]

# gap = np.real(np.mean(gap, axis=-1))

# gap = np.where(tolerance > 1e-10, 0, gap)
# # tolerance = np.mean(tolerance, axis=-1)

df = pd.DataFrame(energy).rename(
    columns={0: "V", 1: "m", 2: "T", 3: "diag", 4: "r", 5: "F"}
)

df['gap'] = gap


# Step 0: Contour of F vs T and mu for multiple r
num_r = len(np.unique(df['r']))
fig, ax = plt.subplots(ncols=num_r, sharey=True, gridspec_kw={'wspace': 0, 'width_ratios': [1, 1]})

N = 500
mu = np.linspace(0, 1.0, N//2)
kvals = 2 * np.cos((2 * np.arange(N) + 1 / (2 * N)))



for r, inner_df in df[['T', 'diag','r','F']].groupby('r'):
    data = np.zeros((mu.size, len(np.unique(inner_df['T']))))

    T = inner_df['T'].to_numpy()
    diag = inner_df['diag'].to_numpy()
    F = inner_df['F'].to_numpy()


    unique_T = np.unique(T)
    unique_diag = np.unique(diag)

    print(unique_diag.shape)

    interp = LinearNDInterpolator(np.array([T, diag]).T, F, fill_value=0.0)

    for idx, muval in enumerate(mu):
        X, Y = np.meshgrid(unique_T, muval + kvals)
        data[idx, :] = np.mean(interp((X, Y)), axis=0)

    x_axis = mu
    y_axis = unique_T

    XX, YY = np.meshgrid(x_axis, y_axis, indexing='ij')
    ZZ = data

    # plt.figure(figsize=(8, 6))

    # print(XX.shape, YY.shape, ZZ.shape)

    norm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

    im = ax[int(r)].pcolormesh(XX, YY, ZZ, cmap='bwr', norm=norm, shading='gouraud')
    # im = ax[int(r)].pcolormesh(X, Y, Z, shading='gouraud')
    ax[int(r)].set_title(f'{int(r)}')

    # plt.pcolormesh(X, Y, Z, shading='gouraud')    # smoother with more levels
    # plt.xlabel('Diagonal value')
    # plt.ylabel('Temperature')
    # plt.title()
    # plt.savefig(f"showplot/DiagvsF{r}.pdf")
ax[0].set_ylabel('Temperature[t]')
fig.supxlabel('Mu')
cbar = fig.colorbar(im, label='Condensation Energy')
# cbar.set_label('Common scale')
plt.tight_layout()

fig.savefig('showplot/FofTvsMu.pdf')
# assert(False)
# Step 1: Contour of F vs T and diag for multiple r
num_r = len(np.unique(df['r']))
fig, ax = plt.subplots(ncols=num_r, sharey=True, gridspec_kw={'wspace': 0})
for r, inner_df in df[['T', 'diag','r','F']].groupby('r'):
    T = inner_df['T'].to_numpy()
    diag = inner_df['diag'].to_numpy()
    F = inner_df['F'].to_numpy()

    xi = np.linspace(diag.min(), diag.max(), 100)
    yi = np.linspace(T.min(), T.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata(
        points=(diag, T),
        values=F,
        xi=(X, Y),
        method='linear'
    )

    # plt.figure(figsize=(8, 6))

    norm = TwoSlopeNorm(vmin=F.min(), vcenter=0, vmax=F.max())

    im = ax[int(r)].pcolormesh(X, Y, Z, cmap='bwr', norm=norm, shading='gouraud')
    # im = ax[int(r)].pcolormesh(X, Y, Z, shading='gouraud')
    ax[int(r)].set_title(f'{int(r)}')

    # plt.pcolormesh(X, Y, Z, shading='gouraud')    # smoother with more levels
    # plt.xlabel('Diagonal value')
    # plt.ylabel('Temperature')
    # plt.title()
    # plt.savefig(f"showplot/DiagvsF{r}.pdf")
ax[0].set_ylabel('Temperature[t]')
fig.supxlabel('Diagonal Value')
cbar = fig.colorbar(im, label='Condensation Energy')
# cbar.set_label('Common scale')
plt.tight_layout()

fig.savefig('showplot/FofTvsD.pdf')

# Step 2: Non-symmetric
num_r = len(np.unique(df['r']))
fig, ax = plt.subplots(ncols=num_r, sharey=True, gridspec_kw={'wspace': 0})
for r, inner_df in df[['T', 'diag','r','F']].groupby('r'):
    T = inner_df['T'].to_numpy()
    diag = inner_df['diag'].to_numpy()
    F = inner_df['F'].to_numpy()

    xi = np.unique(diag)
    xi = xi[xi > 1.1]

    yi = np.unique(T)

    print(xi.size, yi.size)
    # xi = np.linspace(1.0, diag.max(), 100)
    # yi = np.linspace(T.min(), T.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata(
        points=(diag, T),
        values=F,
        xi=(X, Y),
        method='linear'
    )

    # plt.figure(figsize=(8, 6))
    norm = TwoSlopeNorm(vmin=F.min(), vcenter=0, vmax=F.max())

    im = ax[int(r)].pcolormesh(X, Y, Z, cmap='bwr', norm=norm, shading='gouraud')
    ax[int(r)].set_title(f'{int(r)}')

    # plt.pcolormesh(X, Y, Z, shading='gouraud')    # smoother with more levels
    # plt.xlabel('Diagonal value')
    # plt.ylabel('Temperature')
    # plt.title()
    # plt.savefig(f"showplot/DiagvsF{r}.pdf")
ax[0].set_ylabel('Temperature[t]')
fig.supxlabel('Diagonal Value')
cbar = fig.colorbar(im, label='Condensation Energy')
# cbar.set_label('Common scale')
plt.tight_layout()

fig.savefig('showplot/FofTvsDCropped.pdf')

plt.close('all')
# Step 3: Rotated



fig, ax = plt.subplots(ncols=num_r, sharey=True, gridspec_kw={'wspace': 0})
for r, inner_df in df[['T', 'diag','r','F']].groupby('r'):
    T = inner_df['T'].to_numpy()
    diag = inner_df['diag'].to_numpy()
    F = inner_df['F'].to_numpy()

    xi = np.unique(T)
    yi = np.unique(diag)
    yi = yi[yi > 1.1]

    norm = TwoSlopeNorm(vmin=F.min(), vcenter=0, vmax=F.max())


    print(xi.size, yi.size)

    # xi = np.linspace(1.0, diag.max(), 100)
    # yi = np.linspace(T.min(), T.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata(
        points=(T, diag),
        values=F,
        xi=(X, Y),
        method='linear'
    )

    # plt.figure(figsize=(8, 6))

    im = ax[int(r)].pcolormesh(X, Y, Z,cmap='bwr', norm=norm, shading='gouraud')


    # plt.pcolormesh(X, Y, Z, shading='gouraud')    # smoother with more levels
    # plt.xlabel('Diagonal value')
    ax[int(r)].set_title(f'{int(r)}')
    # plt.ylabel('Temperature')
    # plt.title()
    # plt.savefig(f"showplot/DiagvsF{r}.pdf")
ax[0].set_ylabel('Diagonal Value')

ax[0].invert_xaxis()

fig.supxlabel('Temperature[t]')
cbar = fig.colorbar(im, label='Condensation Energy')
# cbar.set_label('Common scale')
plt.tight_layout()
# fig.suptitle('Condensation energy afo T and D')
fig.savefig('showplot/FofTvsDSwapped.pdf')
# Step 4: Curve



temps = [0.0, 0.002, 0.003]

fig, axes = plt.subplots(ncols=num_r, nrows=len(temps), sharey='row', sharex='col',  gridspec_kw={'wspace': 0, 'hspace':0})
for idx, T in enumerate(temps):

    ax = axes[idx]
    for r, inner_df in df[['T', 'diag','r','F']].groupby('r'):

        print(np.abs(inner_df['T'] - T).min())
        inner_df = inner_df[np.abs(inner_df['T'] - T) < 5e-5]
        inner_df = inner_df[inner_df['diag'] >= 1.1]
        diag = inner_df['diag'].to_numpy()
        F = inner_df['F'].to_numpy()
        ax[int(r)].plot(diag, F)

    ax[1].set_ylabel(f'T={T}')
    ax[1].yaxis.set_label_position("right")

axes[0][0].set_title('0')
axes[0][1].set_title('1')


fig.supylabel('Condensation Energy [t]')
fig.supxlabel('Diagonal Value')
# cbar.set_label('Common scale')
# fig.suptitle('Condensation energy afo T and D')
fig.savefig('showplot/FofTvsDSwcurve.pdf')
assert(False)

new_df = df[['T', 'diag', 'r', 'F']]

print(len(np.unique(new_df['T'])))

for T, outer in new_df.groupby(by='T'):

    # if np.max(np.abs(outer['F'])) < 1e-5:
    #     continue
    plt.figure()
    fig, ax = plt.subplots(ncols=2, figsize=(20, 10), sharey=True)
    for num, frame in outer.groupby(by='r'):
        ax[int(num)].plot(frame['diag'], frame['F'])
        ax[int(num)].set_title(f'{num}')

    fig.suptitle(f'T={T}')
    fig.savefig(f'diagplots/T={T}.pdf')
    plt.close('all')

assert(False)

df = df[df['r'] == 1]
# df = df[df['T'] == 0.0]



df['gap'] = np.where(df['F'] <= 0, df['gap'], 0)


# df['F'] = np.where(df['F'] <= 0, df['F'], 0)

print(np.max(gap), np.min(gap))
print(np.max(df['F']), np.min(df['F']))


df = df[['T', 'diag', 'F']]
print(df)
np_df = df.to_numpy()

print(np_df.shape)
points = np_df[:, :-1]
values = np_df[:, -1:]



interp = CloughTocher2DInterpolator(points, values)

mu = 0.1
N = 1000
diag_modes = 0.1 + 2 * np.cos(np.pi *(2 * np.arange(N) + 1)/ (2 * N))



x = np_df[:, 0]
y = np_df[:, 1]
z = np_df[:, 2]

xi = np.linspace(x.min(), 0.01, 100)
yi = np.linspace(y.min(), y.max(), 1000)
X, Y = np.meshgrid(xi, yi)
Z = griddata(
    points=(x, y),
    values=z,
    xi=(X, Y),
    method='linear'
)


plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, shading='gouraud')    # smoother with more levels
plt.colorbar(label='z-value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap of z over (x, y)')
plt.savefig("DiagvsF.pdf")

assert(False)


T = np.linspace(np.min(df['T']), np.max(df['T']), 2*N)
X, Y = np.meshgrid(T, diag_modes, indexing='ij')


res = interp(X, Y)

plt.plot(T, np.mean(res, axis=1).squeeze())
print(res.shape)
plt.savefig('gap.pdf')


assert(False)


print(f'Max tolerance: {np.max(tolerance)}')
# df["tol"] = tolerance

df = df.dropna()

df = df[df['r'] == 0.0]
print(df)
df = df[['T', 'diag', 'F']]
arr = df.to_numpy()
print(arr.shape)





points = arr[:, :-1]
values = arr[:, -1:]

print(points.shape, values.shape)

# print(np.min(values), np.max(values))

# plt.hist(values, bins=100)
# plt.savefig('hist.pdf')
# assert(False)

# print(points.shape, gap.shape)
# assert(False)
interp = NearestNDInterpolator(points, gap[:900], rescale=True)



mu = 0.1
N = 1000

diags = mu + 2 * np.cos(2 * np.pi * np.arange(N) / N)

T = np.linspace(np.min(df['T']), 0.01, 100)


X, Y = np.meshgrid(T, diags, indexing='ij')

res = np.mean(interp(X, Y), axis=1).squeeze()
res = np.mean(res, axis=1)
print(T)
print(res.shape)

plt.figure()
plt.plot(T, res)
plt.savefig("mean.pdf")



# print(df[df['F'].isna()])
# print(df['F'].isna())

# df['gap'] = gap
# other = df[df['F'] > 1e-10]
# # print(other)

# print(0.1 + 2 * np.cos(0.895354))
# df.loc[df['F'] >= 0, 'gap'] = 0

# F = df['F'].to_numpy()


# F[tolerance > 1e-10] = 0
# print(f'Percentage of positive: {100*np.mean(np.where(F > 1e-10, 1, 0))}')


# print(f'Max positive: {np.max(F)}')

# plt.figure()


# N = 100

# kmodes = np.pi * (2 * np.arange(100) + 1) / (2 * 100)

# plt.hist(2 * np.cos(df['kmode']), density=True)
# plt.savefig('newimg2/unique.pdf')

# df["F"] = df["F"].clip(
#     upper=0
# )  # All condensation energies that are positive correspond to 'erroneous' physical state
# # print(df)
# # assert(False)

# df = df.groupby(by=["t", "mu", "V", "m", "r"]).mean().drop(columns=["kmode"])
# # Now, index is T-mu-V-m-r vs F

# # df = df.pivot(index=['t'], columns=['F'])

# groups = df.groupby(by=["mu", "V", "m", "r"])
# print(len(groups))
# # assert(False)


# for elem in groups:
#     fig, (ax, axGap, axTol) = plt.subplots(ncols=3, figsize=(15, 5))


#     ax.set_title("Condensation energy")
#     axGap.set_title("Mean gap")
#     axTol.set_title("Norm")

#     fig.supxlabel("Temperature[t]")

#     X, Y = elem
#     mu, V, m, r = X

#     # print(Y.index)
#     Y = Y.reset_index()
#     # print(Y)
#     # assert(False)
#     F = Y["F"].to_numpy()
#     T = Y["t"].to_numpy()

#     axTol.plot(T, np.log10(Y["tol"].to_numpy()), marker='+')

#     y = Y["gap"].to_numpy()
#     axGap.scatter(T, y, marker='+')
#     axGap.plot(T, y, color='k')
#     # print(Y)
#     # assert(False)

#     meanval = np.mean(np.abs(F))

#     mask = np.abs(F) < 10000000
#     ax.plot(T[mask], F[mask], marker='+')

#     name = str(X)
#     fig.suptitle(f'mu={round(mu, 2)} V={round(V, 2)} m={round(m, 2)} r={r}')
#     fig.savefig("newimg2/" + name + ".pdf")
#     plt.close()
# # print(df)

# # print(df.columns, df.r)

# # [0, 1, 2, 3, 4, 5, 6]
# # [r, free_energy]

# # .groupby(3).sum().get(4)#.to_numpy()

# # T = arr.index.to_numpy()
# # vals = arr.to_numpy()


# # import matplotlib.pyplot as plt
# # plt.plot(T, vals)
# # plt.savefig('condenergy.pdf')


# # print(df)
