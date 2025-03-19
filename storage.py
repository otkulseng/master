import zarr
import jax
import numpy as np
import uuid

import zarr.storage


class Storage:
    def __init__(self, name: str):
        self.store = zarr.storage.LocalStore(Storage.gen_name(name))
        self.root = zarr.open_group(store=self.store)
        self.id = uuid.uuid4().hex

        self.experiment_group: zarr.Group = self.root.create_group(self.id)

    @staticmethod
    def gen_name(name: str):
        return 'data/' + name
    @staticmethod
    def load(name: str):
        return zarr.open_group(store=zarr.storage.LocalStore(Storage.gen_name(name)))

    def save(self, name: str, arr: np.ndarray):

        if name not in self.experiment_group:
            _, N = arr.shape
            self.experiment_group.create_array(name, shape=(0, N), dtype=arr.dtype)

        zarr_array: zarr.Array = self.experiment_group[name]
        zarr_array.append(arr)






_GLOBAL_STORE: Storage = None



def init(name: str):
    global _GLOBAL_STORE
    _GLOBAL_STORE = Storage(name)

def load(name: str) -> zarr.Group:
    return Storage.load(name)

def store(names: list[str], arrs: list[jax.Array]):
    global _GLOBAL_STORE
    for name, arr in zip(names, arrs):
        arr = np.array(arr) # Ensure jax arrays are finished loading

        _GLOBAL_STORE.save(name, arr)


def store_order_params(points: jax.Array, order_params: jax.Array):
    # Wait for jax arrays to be done while  making it possible to enter numpy arrays
    points = np.array(points)
    order_params = np.array(order_params)

    _GLOBAL_STORE.store_order_parameters(points, order_params)
