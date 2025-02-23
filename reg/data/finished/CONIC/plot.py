from pathlib import Path
import yaml
import zarr
import matplotlib.pyplot as plt
import numpy as np

def safe_load_yaml(path: Path) -> tuple[dict, bool]:
    if not path.exists():
        return {}, False

    with open(path) as stream:
        try:
            return yaml.safe_load(stream), True
        except yaml.YAMLError as exc:
            return {}, False

def safe_load_zarr(path: Path) -> tuple[np.ndarray, bool]:
    if not path.exists():
        return None, False

    try:
        return zarr.open(path, mode='r')[:], True
    except:
        return None, False


def filter_func(filter: dict, data: dict) -> bool:
    for key, val in filter.items():
        if key not in data:
            return False

        if filter[key] != data[key]:
            return False
    return True


def main():

# z = zarr.open('result.zarr', mode='r')

# ar = z[:]

# plt.plot(ar)
# plt.savefig('temp.pdf')
    filter = {
        'V': 0.7,
        'kmodes': [101],
        'm': 0.25,
        'mu': 0.1
    }


    results = {}
    xaxis = None
    for folder in Path.cwd().iterdir():


        res, success = safe_load_yaml(folder / 'config.yaml')

        if not success:
            continue

        if not filter_func(filter, res):
            continue


        data, success = safe_load_zarr(folder / 'result.zarr')

        if not success:
            continue

        results[res['Nsc']] = res['Nfm'], data
    # print(results)



    for Nsc, val in results.items():
        x, y = val
        if Nsc not in [110, 120, 130,  150]:
            continue
        plt.plot(x, y / y[0], label=f'{Nsc}')

    plt.legend()
    plt.xlabel(r'$N_{FM}$')
    plt.ylabel(r'$T_c / T_c^0$')
    plt.show()


        # print(conf.exists(), conf.name)


if __name__ == '__main__':
    main()