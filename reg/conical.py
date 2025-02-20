from bodge import *
from hamiltonian import PotentialHamiltonian
import storage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def test(Nsc):
    kmodes = [200]
    mu = 0.1
    pot = 0.7
    m = 0.1
    alpha = 4 * np.pi / 9
    beta = np.pi / 6

    Nfm = np.arange( 3*int(2 * np.pi / beta))

    res = []

    storage.new("CONIC")
    storage.save_kwargs(
        name='config',
        Nsc=Nsc,
        Nfm = Nfm.tolist(),
        kmodes=kmodes,
        mu=mu,
        V = pot,
        m=m,
        alpha=alpha,
        beta=beta
    )

    for n in tqdm(Nfm.tolist()):
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

        T = ham.critical_temperature()
        # ham.solve(0.06238555908203125)
        # ham.crit*

        res.append(T)

        storage.store('result', np.array(res))

    storage.close()

def main():
    test(100)
    test(120)
    test(140)
    test(125)
    test(130)
    test(135)



if __name__ == '__main__':
    main()

