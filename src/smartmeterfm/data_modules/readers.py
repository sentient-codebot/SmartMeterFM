"""HDF5 readers for WPuQ datasets."""

import numpy as np


class BaseHDF5Reader:
    """Base class for HDF5 readers with context manager + iterator protocol."""

    def __init__(self, filename: str, column_names: list[str]):
        self.filename = filename
        self.column_names = column_names
        self.f = None
        self._items: list = []
        self._index: int = 0

    def __enter__(self):
        import h5py

        self.f = h5py.File(self.filename, "r")
        self._items = self._discover_items()
        return self

    def __exit__(self, *args):
        self.f.close()

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._items):
            raise StopIteration
        result = self._read_item(self._items[self._index])
        self._index += 1
        return result

    def __len__(self):
        return len(self._items)

    def _discover_items(self) -> list:
        """Return list of items to iterate over. Called in __enter__."""
        raise NotImplementedError

    def _read_item(self, item) -> np.ndarray:
        """Read and return one item as a structured array."""
        raise NotImplementedError


class WPuQReader(BaseHDF5Reader):
    """Read heat pump data from HDF5 (only NO_PV group)."""

    def _discover_items(self) -> list[str]:
        sfhs = list(self.f["NO_PV"].keys())
        return [sfh for sfh in sfhs if "HEATPUMP" in self.f["NO_PV"][sfh].keys()]

    def _read_item(self, sfh: str) -> np.ndarray:
        table = self.f["NO_PV"][sfh]["HEATPUMP"]["table"]
        table = np.array(table)
        return table[self.column_names]


class WPuQHouseholdReader(BaseHDF5Reader):
    """Read household electricity data from all houses (NO_PV + WITH_PV)."""

    def _discover_items(self) -> list[tuple[str, str]]:
        items = []
        for group in ["NO_PV", "WITH_PV"]:
            if group in self.f:
                for sfh in self.f[group].keys():
                    if "HOUSEHOLD" in self.f[group][sfh].keys():
                        items.append((group, sfh))
        return items

    def _read_item(self, item: tuple[str, str]) -> np.ndarray:
        group, sfh = item
        table = self.f[group][sfh]["HOUSEHOLD"]["table"]
        table = np.array(table)
        return table[self.column_names]


class WPuQPVReader(BaseHDF5Reader):
    """Read PV generation data from HDF5 (MISC/PV1/PV/INVERTER)."""

    directions = ["EAST", "WEST", "SOUTH"]

    def _discover_items(self) -> list[int]:
        return [0]  # single logical item: all directions concatenated

    def _read_item(self, _item: int) -> np.ndarray:
        tables = []
        for direction in self.directions:
            table = self.f["MISC"]["PV1"]["PV"]["INVERTER"][direction]["table"]
            table = np.array(table)
            direction_col = np.full((table.shape[0],), direction, dtype="U10")
            new_dtype = np.dtype(table.dtype.descr + [("DIRECTION", "U10")])
            new_table = np.empty(table.shape, dtype=new_dtype)
            for field in table.dtype.names:
                new_table[field] = table[field]
            new_table["DIRECTION"] = direction_col
            tables.append(new_table[self.column_names])
        return np.concatenate(tables, axis=0)

    def __len__(self):
        return len(self.directions)
