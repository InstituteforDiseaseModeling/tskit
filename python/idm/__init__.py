import _idm
import numpy as np


align_data = _idm.align_data

def get_genomes(ts):
    if not hasattr(ts, "_ll_tree_sequence"):
        raise RuntimeError("argument must be a tskit.TreeSequence")
    return _idm.get_genomes(ts._ll_tree_sequence)


def calculate_ibx(genome, indices=None, intervals=None):
    return _idm.calculate_ibx(genome, indices, intervals)


class IbxResults:

    def __init__(self, genome, indices=None, intervals=None):

        self._indices, self._intervals = IbxResults._validate_inputs(genome, indices, intervals)

        self._genome = genome
        self._ibx, self._hashes, self._mapping = _idm.calculate_ibx(self._genome, self._indices, self._intervals)

        return

    @staticmethod
    def _validate_inputs(genome, indices, intervals):

        if not isinstance(genome, np.ndarray):
            raise RuntimeError("Genome must be a NumPy array.")

        if len(genome.shape) != 2:
            raise RuntimeError("Genome must be a two dimensional array.")

        if genome.dtype not in set([np.dtype("int8"), np.dtype("uint8"), np.dtype("int16"), np.dtype("uint16"), np.dtype("int32"), np.dtype("uint32")]):
            raise RuntimeError("Genome must be of an integral type with size 1, 2, or 4 bytes.")

        if indices is None:
            ids = np.asarray(range(genome.shape[0]), dtype=np.uint32)
        else:
            ids = np.asarray(indices, dtype=np.uint32)

            if len(ids.shape) != 1:
                raise RuntimeError("Indices must be a vector (one dimensional) array.")


        if intervals is None:
            ints = np.ones(genome.shape[1], dtype=np.uint32)
        else:
            ints = np.asarray(intervals, dtype=np.uint32)

            if len(ints.shape) != 1:
                raise RuntimeError("Intervals must be a vector (one dimensional) array.")

        return ids, ints

    @property
    def genome(self):
        return self._genome

    @property
    def indices(self):
        return self._indices

    @property
    def intervals(self):
        return self._intervals

    @property
    def pairs(self):
        return self._ibx

    @property
    def hashes(self):
        return self._hashes

    @property
    def mapping(self):
        return self._mapping

    def __getitem__(self, key):

        if not isinstance(key, (slice, tuple)):
            raise TypeError(f"results indices must be slices or tuples, not {type(key).__name__}")

        if isinstance(key, slice):
            if key.start is None or key.step is not None:
                raise RuntimeError("results indices must be of the form [x:y] or [x,y]")
            ida, idb = key.start, key.stop
        else:   # tuple
            if len(key) != 2:
                raise RuntimeError("results indices must be of the form [x:y] or [x,y]")
            ida, idb = key

        try:
            indexa = np.where(self._indices == ida)[0][0]
        except IndexError:
            raise IndexError(f"genome id {ida} is not in the list of indices")
        try:
            indexb = np.where(self._indices == idb)[0][0]
        except IndexError:
            raise IndexError(f"genome id {idb} is not in the list of indices")

        hasha = self._hashes[indexa]
        hashb = self._hashes[indexb]
        ibxa = self._mapping[hasha]
        ibxb = self._mapping[hashb]

        return self._ibx[ibxa, ibxb]
