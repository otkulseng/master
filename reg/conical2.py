from bodge import *
from hamiltonian import PotentialHamiltonian
import storage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

def test(Nsc, Nfm: list):
    kmodes = [101]
    mu = 0.1
    pot = 1.0
    m = 0.25
    alpha = 4 * np.pi / 9
    beta = np.pi / 6
    # kmodes = [101]
    # mu = 0.1
    # pot = 0.7
    # m = 0.25
    # alpha = 4 * np.pi / 9
    # beta = np.pi / 6


    storage.new("CONIC")
    storage.save_kwargs(
        name='config',
        Nsc=Nsc,
        Nfm = Nfm,
        kmodes=kmodes,
        mu=mu,
        V = pot,
        m=m,
        alpha=alpha,
        beta=beta
    )

    if 0 not in Nfm:
        Nfm.append(0)

    hams = {}

    for n in Nfm:
        shape = (int(Nsc + n), 1, 1)


        lat = CubicLattice(shape)
        ham = PotentialHamiltonian(lat, kmodes=kmodes)
        with ham as (H, V):
            for i in lat.sites():
                x, _, _ = i

                angle = beta * x
                vec = np.array([np.cos(alpha), np.sin(alpha) * np.cos(angle), np.sin(alpha)*np.sin(beta)])


                matr = - mu * sigma0

                if x < Nsc:
                    V[i, i] = - pot
                else:
                    matr += m * (vec[0] * sigma1 + vec[1] * sigma2 + vec[2]*sigma3)

                H[i, i] = matr

            for i, j in lat.bonds():
                H[i, j] = -1.0 * sigma0

        hams[n] = ham
        # tmax = 1
        # if len(res) > 0:
        #     tmax = res[0]
        # T = ham.solver().critical_temperature(max_temp=tmax)

        # res.append(T)

        # storage.store('result', np.array(res))
    # t0 = hams[0].solver().critical_temperature(max_temp = 0.01)
    t0 = hams[0].solver().critical_temperature(max_temp = 1.0)

    temps = torch.linspace(0, t0, 20)
    storage.store('temps', temps.numpy())
    for Nfm in Nfm:
        ham = hams[Nfm].solver()

        result = []
        for T in temps:
            result.append(
                ham.solve_integral(T).numpy()
            )
            storage.store(f'result-{Nfm}', np.array(result))

        plt.plot(temps, result[0].numpy())
        plt.show()

    storage.close()

def main():
    test(50, [6, 15])
    # test(125)
    # test(130)
    # test(135)



if __name__ == '__main__':
    main()

