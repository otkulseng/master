from pathlib import Path
import zarr
import time
from typing import Union
import torch
import numpy as np

import yaml

class Storage():
    def __init__(self, name, create_run=True):
        # Create new


        self.root = Path("data") / "temp" / name

        if create_run:
            run = str(time.time_ns())
            self.root = self.root / run
            self.root.mkdir(exist_ok=True, parents=True)

        self.arrays = {}
    def save(self, name: str, arr: np.ndarray):
        if '.zarr' not in name:
            name = f'{name}.zarr'

        self.arrays[name] = zarr.create_array(
            store = self.root / name,
            data = arr,
            overwrite=True
        )
    def save_kwargs(self, name: str, *args, **kwargs):
        if '.yaml' not in name:
            name = f'{name}.yaml'

        with open(self.root / name, 'w') as f:
            yaml.dump(dict(kwargs), f)


    def done(self):
        new_path = Path("data/finished") / self.root.relative_to("data/temp")

        new_path.mkdir(parents=True, exist_ok=True)
        self.root.rename(new_path)

STORE = None

def new(name: str):
    global STORE
    STORE = Storage(name)

def open_(name: str):
    global STORE
    STORE = Storage(name, create_run=False)

def store(name: str, arr: Union[np.ndarray, torch.Tensor]):
    global STORE
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()

    STORE.save(name, arr)

def save_kwargs(name: str, *args, **kwargs):
    global STORE
    STORE.save_kwargs(name, **kwargs)

def close():
    global STORE
    STORE.done()



