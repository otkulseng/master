import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt




def order(n):
    def func(x, *args):
        total = args[0]
        for i in range(1, n, 2):
            total += args[i] / (args[i+1] + x**2)
        return total
    return func

xdata = np.linspace(-1, 1, 101)

def gen_func(temperature):
    def free_energy_func(x):
        norm = 100
        abs_term =  norm * np.abs(x) / 2
        if temperature > 0:
            abs_term += temperature * np.log(1 + np.exp(- norm / temperature * np.abs(x)))
        return abs_term
    return free_energy_func

true_func = gen_func(0.1)
ydata = true_func(xdata)


p0 = np.array([])
for ord in [12, 24]:
    func = order(ord)
    print(ord)

    p0 = np.ones(ord + 1)


    popt, pcov = curve_fit(func, xdata, ydata, p0=list(p0), bounds=(-10000, 10000), maxfev=1000000)
    p0 = np.array(popt)
    print(p0)
    if ord < 8:
        continue
    plt.plot(xdata, np.log10(np.abs(np.abs(func(xdata, *popt)) - true_func(xdata))), label=f"{ord}")
    # plt.plot(xdata,func(xdata, *popt) , label=f"{ord}")

# plt.plot(xdata, np.abs(xdata), label="True")
plt.legend()
print(popt)
plt.savefig("scaling2.pdf")