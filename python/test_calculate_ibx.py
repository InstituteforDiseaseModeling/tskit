#! /usr/bin/env python3

from hashlib import sha256
import numpy as np
from pathlib import Path
import unittest

import tskit
import _idm

NUM_GENOMES = 1024
NUM_INTERVALS = 65536
NUM_ROOTS = 8
prng = np.random.default_rng()
gen08 = prng.integers(0, high=NUM_ROOTS, size=(NUM_GENOMES, NUM_INTERVALS), dtype=np.uint8)
gen16 = prng.integers(0, high=NUM_ROOTS, size=(NUM_GENOMES, NUM_INTERVALS), dtype=np.uint16)
gen32 = prng.integers(0, high=NUM_ROOTS, size=(NUM_GENOMES, NUM_INTERVALS), dtype=np.uint32)

class TestIdm(unittest.TestCase):

    def test_simple08(self):

        NUM_INDICES = 12
        indices = np.asarray(range(NUM_INDICES), dtype=np.uint32)

        ibx, hashes, mapping = _idm.calculate_ibx(gen08, indices)

        self.assertEqual(ibx.shape, (NUM_INDICES, NUM_INDICES))
        self.assertEqual(len(hashes), NUM_INDICES)
        self.assertIsNotNone(mapping)
        for g in range(NUM_INDICES):
            self.assertEqual(hashes[g], sha256(gen08[g,:]).hexdigest())
        for a in range(NUM_INDICES):
            ai = mapping[hashes[a]]
            for b in range(NUM_INDICES):
                bi = mapping[hashes[b]]
                self.assertEqual(ibx[ai,bi], np.sum(gen08[a,:]==gen08[b,:]))
                self.assertEqual(ibx[ai,bi], ibx[bi,ai])

        return

    def test_subset08(self):

        NUM_INDICES = 12
        prng = np.random.default_rng()
        indices = prng.choice(np.asarray(range(NUM_GENOMES), dtype=np.uint32), size=NUM_INDICES, replace=False)
        indices.sort()
        print(f"{indices=}")

        ibx, hashes, mapping = _idm.calculate_ibx(gen08, indices)

        self.assertEqual(ibx.shape, (NUM_INDICES, NUM_INDICES))
        self.assertEqual(len(hashes), NUM_INDICES)
        self.assertIsNotNone(mapping)
        for index in range(NUM_INDICES):
            g = indices[index]
            self.assertEqual(hashes[index], sha256(gen08[g,:]).hexdigest())
        for i in range(NUM_INDICES):        # access monotonically...
            gi = indices[i]                 # actual genome id = indices[i]
            hi = mapping[hashes[i]]         # hash of this genome is at hashes[i], index of this hash in the unique list is mapping[hashes[i]]
            for j in range(NUM_INDICES):    # access monotonically...
                gj = indices[j]             # actual genome id = indices[j]
                hj = mapping[hashes[j]]     # hash of this genome is at hashes[j], index of this hash in the unique list is mapping[hashes[j]]
                # See if the value in ibx[hi,hj] == the calculated IBx of genome[gi] and genome[gj]...
                self.assertEqual(ibx[hi,hj], np.sum(gen08[gi,:]==gen08[gj,:]))
                # Make sure the matrix is symmetric.
                self.assertEqual(ibx[hi,hj], ibx[hj,hi])

        return

    def test_with_intervals08(self):

        NUM_INDICES = 12
        prng = np.random.default_rng()
        indices = prng.choice(np.asarray(range(NUM_GENOMES), dtype=np.uint32), size=NUM_INDICES, replace=False)
        indices.sort()
        print(f"{indices=}")

        ibx1, hashes1, mapping1 = _idm.calculate_ibx(gen08, indices)     # intervals == 1 by default

        intervals = np.ones(NUM_INTERVALS, dtype=np.uint32) * 2
        ibx2, hashes2, mapping2 = _idm.calculate_ibx(gen08, indices, intervals)

        self.assertTupleEqual(ibx2.shape, ibx1.shape)
        self.assertListEqual(hashes2, hashes1)
        self.assertDictEqual(mapping2, mapping1)

        rows, columns = ibx1.shape
        for row in range(rows):
            for column in range(columns):
                self.assertEqual(ibx2[row, column], 2*ibx1[row, column])

        return

    def test_simple16(self):

        NUM_INDICES = 12
        indices = np.asarray(range(NUM_INDICES), dtype=np.uint32)

        ibx, hashes, mapping = _idm.calculate_ibx(gen16, indices)

        self.assertEqual(ibx.shape, (NUM_INDICES, NUM_INDICES))
        self.assertEqual(len(hashes), NUM_INDICES)
        self.assertIsNotNone(mapping)
        for g in range(NUM_INDICES):
            self.assertEqual(hashes[g], sha256(gen16[g,:]).hexdigest())
        for a in range(NUM_INDICES):
            ai = mapping[hashes[a]]
            for b in range(NUM_INDICES):
                bi = mapping[hashes[b]]
                self.assertEqual(ibx[ai,bi], np.sum(gen16[a,:]==gen16[b,:]))
                self.assertEqual(ibx[ai,bi], ibx[bi,ai])

        return

    def test_subset16(self):

        NUM_INDICES = 12
        prng = np.random.default_rng()
        indices = prng.choice(np.asarray(range(NUM_GENOMES), dtype=np.uint32), size=NUM_INDICES, replace=False)
        indices.sort()
        print(f"{indices=}")

        ibx, hashes, mapping = _idm.calculate_ibx(gen16, indices)

        self.assertEqual(ibx.shape, (NUM_INDICES, NUM_INDICES))
        self.assertEqual(len(hashes), NUM_INDICES)
        self.assertIsNotNone(mapping)
        for index in range(NUM_INDICES):
            g = indices[index]
            self.assertEqual(hashes[index], sha256(gen16[g,:]).hexdigest())
        for i in range(NUM_INDICES):        # access monotonically...
            gi = indices[i]                 # actual genome id = indices[i]
            hi = mapping[hashes[i]]         # hash of this genome is at hashes[i], index of this hash in the unique list is mapping[hashes[i]]
            for j in range(NUM_INDICES):    # access monotonically...
                gj = indices[j]             # actual genome id = indices[j]
                hj = mapping[hashes[j]]     # hash of this genome is at hashes[j], index of this hash in the unique list is mapping[hashes[j]]
                # See if the value in ibx[hi,hj] == the calculated IBx of genome[gi] and genome[gj]...
                self.assertEqual(ibx[hi,hj], np.sum(gen16[gi,:]==gen16[gj,:]))
                # Make sure the matrix is symmetric.
                self.assertEqual(ibx[hi,hj], ibx[hj,hi])

        return

    def test_with_intervals16(self):

        NUM_INDICES = 12
        prng = np.random.default_rng()
        indices = prng.choice(np.asarray(range(NUM_GENOMES), dtype=np.uint32), size=NUM_INDICES, replace=False)
        indices.sort()
        print(f"{indices=}")

        ibx1, hashes1, mapping1 = _idm.calculate_ibx(gen16, indices)     # intervals == 1 by default

        intervals = np.ones(NUM_INTERVALS, dtype=np.uint32) * 2
        ibx2, hashes2, mapping2 = _idm.calculate_ibx(gen16, indices, intervals)

        self.assertTupleEqual(ibx2.shape, ibx1.shape)
        self.assertListEqual(hashes2, hashes1)
        self.assertDictEqual(mapping2, mapping1)

        rows, columns = ibx1.shape
        for row in range(rows):
            for column in range(columns):
                self.assertEqual(ibx2[row, column], 2*ibx1[row, column])

        return

    def test_simple32(self):

        NUM_INDICES = 12
        indices = np.asarray(range(NUM_INDICES), dtype=np.uint32)

        ibx, hashes, mapping = _idm.calculate_ibx(gen32, indices)

        self.assertEqual(ibx.shape, (NUM_INDICES, NUM_INDICES))
        self.assertEqual(len(hashes), NUM_INDICES)
        self.assertIsNotNone(mapping)
        for g in range(NUM_INDICES):
            self.assertEqual(hashes[g], sha256(gen32[g,:]).hexdigest())
        for a in range(NUM_INDICES):
            ai = mapping[hashes[a]]
            for b in range(NUM_INDICES):
                bi = mapping[hashes[b]]
                self.assertEqual(ibx[ai,bi], np.sum(gen32[a,:]==gen32[b,:]))
                self.assertEqual(ibx[ai,bi], ibx[bi,ai])

        return

    def test_subset32(self):

        NUM_INDICES = 12
        prng = np.random.default_rng()
        indices = prng.choice(np.asarray(range(NUM_GENOMES), dtype=np.uint32), size=NUM_INDICES, replace=False)
        indices.sort()
        print(f"{indices=}")

        ibx, hashes, mapping = _idm.calculate_ibx(gen32, indices)

        self.assertEqual(ibx.shape, (NUM_INDICES, NUM_INDICES))
        self.assertEqual(len(hashes), NUM_INDICES)
        self.assertIsNotNone(mapping)
        for index in range(NUM_INDICES):
            g = indices[index]
            self.assertEqual(hashes[index], sha256(gen32[g,:]).hexdigest())
        for i in range(NUM_INDICES):        # access monotonically...
            gi = indices[i]                 # actual genome id = indices[i]
            hi = mapping[hashes[i]]         # hash of this genome is at hashes[i], index of this hash in the unique list is mapping[hashes[i]]
            for j in range(NUM_INDICES):    # access monotonically...
                gj = indices[j]             # actual genome id = indices[j]
                hj = mapping[hashes[j]]     # hash of this genome is at hashes[j], index of this hash in the unique list is mapping[hashes[j]]
                # See if the value in ibx[hi,hj] == the calculated IBx of genome[gi] and genome[gj]...
                self.assertEqual(ibx[hi,hj], np.sum(gen32[gi,:]==gen32[gj,:]))
                # Make sure the matrix is symmetric.
                self.assertEqual(ibx[hi,hj], ibx[hj,hi])

        return

    def test_with_intervals32(self):

        NUM_INDICES = 12
        prng = np.random.default_rng()
        indices = prng.choice(np.asarray(range(NUM_GENOMES), dtype=np.uint32), size=NUM_INDICES, replace=False)
        indices.sort()
        print(f"{indices=}")

        ibx1, hashes1, mapping1 = _idm.calculate_ibx(gen32, indices)     # intervals == 1 by default

        intervals = np.ones(NUM_INTERVALS, dtype=np.uint32) * 2
        ibx2, hashes2, mapping2 = _idm.calculate_ibx(gen32, indices, intervals)

        self.assertTupleEqual(ibx2.shape, ibx1.shape)
        self.assertListEqual(hashes2, hashes1)
        self.assertDictEqual(mapping2, mapping1)

        rows, columns = ibx1.shape
        for row in range(rows):
            for column in range(columns):
                self.assertEqual(ibx2[row, column], 2*ibx1[row, column])

        return

    def test_workflow1(self):

        ts = tskit.load(Path(__file__).parent.absolute() / "tree-sequence.ts")
        genomes = _idm.get_genomes(ts._ll_tree_sequence)
        ibx, hashes, mapping = _idm.calculate_ibx(genomes)
        prng = np.random.default_rng()
        NUM_INDICES = 12
        testids = prng.choice(np.asarray(range(genomes.shape[0]), dtype=np.uint32), size=NUM_INDICES, replace=False)
        for row in testids:
            irow = mapping[hashes[row]]
            for col in testids:
                icol = mapping[hashes[col]]
                self.assertEqual(ibx[irow,icol], np.sum(genomes[row,:]==genomes[col,:]))
                self.assertEqual(ibx[icol,irow], ibx[irow,icol])

        return

    def test_workflow2(self):

        tc = tskit.load(Path(__file__).parent.absolute() / "tree-collection.ts")
        genomes = _idm.get_genomes(tc._ll_tree_sequence)
        ibx, hashes, mapping = _idm.calculate_ibx(genomes)
        prng = np.random.default_rng()
        NUM_INDICES = 12
        testids = prng.choice(np.asarray(range(genomes.shape[0]), dtype=np.uint32), size=NUM_INDICES, replace=False)
        for row in testids:
            irow = mapping[hashes[row]]
            for col in testids:
                icol = mapping[hashes[col]]
                self.assertEqual(ibx[irow,icol], np.sum(genomes[row,:]==genomes[col,:]))
                self.assertEqual(ibx[icol,irow], ibx[irow,icol])

        return


if __name__ == "__main__":
    unittest.main()
