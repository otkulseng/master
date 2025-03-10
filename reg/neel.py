from bodge import *
from hamiltonian import PotentialHamiltonian
import storage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

def gen_hamilt(N, mu, m, pot, r, kmodes):
    shape = (1, 4, N)
    lat = CubicLattice(shape)
    ham = PotentialHamiltonian(lat, kmodes=kmodes)
    with ham as (H, V):
        for i in lat.sites():

            _, y, z = i
            mval = 0
            # mag_matr = m * sigma3

            if y == 1 or y == 2:
                V[i, i] = -pot
                mval = 0
            elif y == 0:
                mval = 1
            else:
                mval = 1 if z < r else -1
            H[i, i] = - mu * sigma0 + mval * m * sigma3

        for i, j in lat.bonds():
            H[i, j] = -1.0 * sigma0

        for i, j in lat.edges(axis=2):
            H[i, j] = -1.0 * sigma0
    return ham

def test(N, mu, m, pot):
    kmodes = [201]


    storage.new("NEEL")
    storage.save_kwargs(
        name='config',
        N=N,
        mu = m,
        V= pot,
        # r=r
    )

    res = {}

    T = torch.linspace(0, 0.02, N)
    # T = torch.linspace(0, 0.1, N)
    # solver = gen_hamilt(N, mu, m, pot, 0, kmodes).solver()
    # x0 = solver.solve_integral(torch.tensor(0.05))
    # print(x0.mean(dim=0).real)
    # assert(False)
    # This

    for r in tqdm(range(N+1)):
        for t in T:
            try:
                solver = gen_hamilt(N, mu, m, pot, r, kmodes).solver()
                xx = solver.solve_integral(t)
                storage.store(f'{r}-{t}', xx.numpy())
                cond = solver.condensation_energy(xx, t)

                cur = res.get(str(r), [])
                cur = cur + [(t.item(), cond.item().real)]
                res[str(r)] = cur
                storage.save_kwargs("res", **res)
            except Exception as e:
                print(e)
                continue

    storage.close()

def main():
    test(25, 0.1, 0.1, 0.9)
    # test(25, 0.05, 0.1, 0.9)
    # test(25, 0.1, 0.1, 0.85)
    # test(25, 0.05, 0.1, 0.85)
    # test(125)
    # test(130)
    # test(135)



if __name__ == '__main__':
    main()

