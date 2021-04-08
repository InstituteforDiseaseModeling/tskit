#! /usr/bin/env python3

from hashlib import sha256
import numpy as np
from pathlib import Path
import unittest

import tskit
import idm

NUM_GENOMES = 1024
NUM_INTERVALS = 65536
NUM_ROOTS = 8
prng = np.random.default_rng()
gen08 = idm.align_data(prng.integers(0, high=NUM_ROOTS, size=(NUM_GENOMES, NUM_INTERVALS), dtype=np.uint8))
gen16 = idm.align_data(prng.integers(0, high=NUM_ROOTS, size=(NUM_GENOMES, NUM_INTERVALS), dtype=np.uint16))
gen32 = idm.align_data(prng.integers(0, high=NUM_ROOTS, size=(NUM_GENOMES, NUM_INTERVALS), dtype=np.uint32))

class TestIdm(unittest.TestCase):

    def test_simple08(self):

        NUM_INDICES = 12
        indices = np.asarray(range(NUM_INDICES), dtype=np.uint32)

        ibx, hashes, mapping = idm.calculate_ibx(gen08, indices)

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

        ibx, hashes, mapping = idm.calculate_ibx(gen08, indices)

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

        ibx1, hashes1, mapping1 = idm.calculate_ibx(gen08, indices)     # intervals == 1 by default

        intervals = np.ones(NUM_INTERVALS, dtype=np.uint32) * 2
        ibx2, hashes2, mapping2 = idm.calculate_ibx(gen08, indices, intervals)

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

        ibx, hashes, mapping = idm.calculate_ibx(gen16, indices)

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

        ibx, hashes, mapping = idm.calculate_ibx(gen16, indices)

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

        ibx1, hashes1, mapping1 = idm.calculate_ibx(gen16, indices)     # intervals == 1 by default

        intervals = np.ones(NUM_INTERVALS, dtype=np.uint32) * 2
        ibx2, hashes2, mapping2 = idm.calculate_ibx(gen16, indices, intervals)

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

        ibx, hashes, mapping = idm.calculate_ibx(gen32, indices)

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

        ibx, hashes, mapping = idm.calculate_ibx(gen32, indices)

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

        ibx1, hashes1, mapping1 = idm.calculate_ibx(gen32, indices)     # intervals == 1 by default

        intervals = np.ones(NUM_INTERVALS, dtype=np.uint32) * 2
        ibx2, hashes2, mapping2 = idm.calculate_ibx(gen32, indices, intervals)

        self.assertTupleEqual(ibx2.shape, ibx1.shape)
        self.assertListEqual(hashes2, hashes1)
        self.assertDictEqual(mapping2, mapping1)

        rows, columns = ibx1.shape
        for row in range(rows):
            for column in range(columns):
                self.assertEqual(ibx2[row, column], 2*ibx1[row, column])

        return

    def test_workflow1(self):

        ts = tskit.load(Path(__file__).parent.absolute() / "data" / "idm" / "tree-sequence.ts")
        genomes, intervals = idm.get_genomes(ts)
        ibx, hashes, mapping = idm.calculate_ibx(genomes)
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

        tc = tskit.load(Path(__file__).parent.absolute() / "data" / "idm" / "tree-collection.ts")
        genomes, intervals = idm.get_genomes(tc)
        ibx, hashes, mapping = idm.calculate_ibx(genomes)
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


class TestIbxResults(unittest.TestCase):

    genomes = None
    intervals = None

    @classmethod
    def setUpClass(cls):
        ts = tskit.load(Path(__file__).parent.absolute() / "data" / "idm" / "tree-sequence.ts")
        TestIbxResults.genomes, TestIbxResults.intervals = idm.get_genomes(ts)
        return

    def test_happy_path_no_indices_no_intervals(self):
        results = idm.IbxResults(TestIbxResults.genomes)
        self.assertEqual(results[0,0], 147)     # identity = total length
        self.assertEqual(results[0,1], 0)       # these are separate roots
        self.assertEqual(results[0,4], 73)      # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,4], 74)      # 4 is a recombination of 0 and 2
        return

    def test_happy_path_yes_indices_no_intervals(self):
        indices = np.asarray([1, 2, 4, 5], dtype=np.uint32)
        results = idm.IbxResults(TestIbxResults.genomes, indices=indices)
        self.assertEqual(results.pairs.shape, (4, 4))
        self.assertEqual(results[1,1], 147)     # identity = total length
        self.assertEqual(results[1,2], 0)       # 1 and 2 are separate roots
        self.assertEqual(results[2,4], 74)      # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,5], 66)      # 5 is a recombination of 0 and 2
        return

    def test_happy_path_no_indices_yes_intervals(self):
        results = idm.IbxResults(TestIbxResults.genomes, intervals=TestIbxResults.intervals)
        self.assertEqual(results[0,0], 14000)   # identity = total length
        self.assertEqual(results[0,1], 0)       # these are separate roots
        self.assertEqual(results[0,4], 7204)    # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,4], 6796)    # 4 is a recombination of 0 and 2
        return

    def test_happy_path_yes_indices_yes_intervals(self):
        indices = np.asarray([1, 2, 4, 5], dtype=np.uint32)
        results = idm.IbxResults(TestIbxResults.genomes, indices=indices, intervals=TestIbxResults.intervals)
        self.assertEqual(results.pairs.shape, (4, 4))
        self.assertEqual(results[1,1], 14000)   # identity = total length
        self.assertEqual(results[1,2], 0)       # 1 and 2 are separate roots
        self.assertEqual(results[2,4], 6796)    # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,5], 6423)    # 5 (indices[3]) is a recombination of 0 and 2
        return

    def test_with_indices_list(self):
        results = idm.IbxResults(TestIbxResults.genomes, indices=[1, 2, 4, 5])
        self.assertEqual(results.pairs.shape, (4, 4))
        self.assertEqual(results[1,1], 147)     # identity = total length
        self.assertEqual(results[1,2], 0)       # 1 and 2 are separate roots
        self.assertEqual(results[2,4], 74)      # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,5], 66)      # 5 (indices[3]) is a recombination of 0 and 2
        return

    def test_with_indices_tuple(self):
        results = idm.IbxResults(TestIbxResults.genomes, indices=(1, 2, 4, 5))
        self.assertEqual(results.pairs.shape, (4, 4))
        self.assertEqual(results[1,1], 147)     # identity = total length
        self.assertEqual(results[1,2], 0)       # 1 and 2 are separate roots
        self.assertEqual(results[2,4], 74)      # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,5], 66)      # 5 (indices[3]) is a recombination of 0 and 2
        return

    def test_with_indices_float(self):
        indices = np.asarray([1, 2, 4, 5], dtype=np.float64)
        results = idm.IbxResults(TestIbxResults.genomes, indices=indices)
        self.assertEqual(results.pairs.shape, (4, 4))
        self.assertEqual(results[1,1], 147)     # identity = total length
        self.assertEqual(results[1,2], 0)       # 1 and 2 are separate roots
        self.assertEqual(results[2,4], 74)      # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,5], 66)      # 5 (indices[3]) is a recombination of 0 and 2
        return

    def test_with_intervals_list(self):
        intervals = list(TestIbxResults.intervals)
        results = idm.IbxResults(TestIbxResults.genomes, intervals=intervals)
        self.assertEqual(results[0,0], 14000)   # identity = total length
        self.assertEqual(results[0,1], 0)       # these are separate roots
        self.assertEqual(results[0,4], 7204)    # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,4], 6796)    # 4 is a recombination of 0 and 2
        return

    def test_with_intervals_tuple(self):
        intervals = tuple(TestIbxResults.intervals)
        results = idm.IbxResults(TestIbxResults.genomes, intervals=intervals)
        self.assertEqual(results[0,0], 14000)   # identity = total length
        self.assertEqual(results[0,1], 0)       # these are separate roots
        self.assertEqual(results[0,4], 7204)    # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,4], 6796)    # 4 is a recombination of 0 and 2
        return

    def test_with_intervals_float(self):
        intervals = np.array(TestIbxResults.intervals, dtype=np.float64)
        results = idm.IbxResults(TestIbxResults.genomes, intervals=intervals)
        self.assertEqual(results[0,0], 14000)   # identity = total length
        self.assertEqual(results[0,1], 0)       # these are separate roots
        self.assertEqual(results[0,4], 7204)    # 4 is a recombination of 0 and 2
        self.assertEqual(results[2,4], 6796)    # 4 is a recombination of 0 and 2
        return

    def test_bad_genome_type(self):
        with self.assertRaises(RuntimeError):
            _ = idm.IbxResults([[0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1]])
        return

    def test_bad_genome_shape(self):
        prng = np.random.default_rng()
        threed = prng.integers(0, 16, (8, 16, 32), dtype=np.uint32)
        with self.assertRaises(RuntimeError):
            _ = idm.IbxResults(threed)
        return

    def test_bad_genome_dtype(self):
        doubles = np.array(TestIbxResults.genomes, dtype=np.float64)
        with self.assertRaises(RuntimeError):
            _ = idm.IbxResults(doubles)
        return

    def test_bad_indices_shape(self):
        prng = np.random.default_rng()
        indices = prng.integers(1, 1024, (147, 147), dtype=np.uint32)
        with self.assertRaises(RuntimeError):
            _ = idm.IbxResults(TestIbxResults.genomes, indices=indices)
        return

    def test_bad_intervals_shape(self):
        prng = np.random.default_rng()
        intervals = prng.integers(1, 1024, (147, 147), dtype=np.uint32)
        with self.assertRaises(RuntimeError):
            _ = idm.IbxResults(TestIbxResults.genomes, intervals=intervals)
        return

    def test_get_item_with_slice(self):
        results = idm.IbxResults(TestIbxResults.genomes)
        self.assertEqual(results[0:0], 147)     # identity = total length
        self.assertEqual(results[0:1], 0)       # these are separate roots
        self.assertEqual(results[0:4], 73)      # 4 is a recombination of 0 and 2
        self.assertEqual(results[2:4], 74)      # 4 is a recombination of 0 and 2
        return

    # def test_get_item_with_tuple(self):
        # tested in happy path tests
        # return

    def test_get_item_with_bad_indexa(self):
        results = idm.IbxResults(TestIbxResults.genomes)
        with self.assertRaises(IndexError):
            _ = results[42,0]
        return

    def test_get_item_width_bad_indexb(self):
        results = idm.IbxResults(TestIbxResults.genomes)
        with self.assertRaises(IndexError):
            _ = results[0,42]
        return


if __name__ == "__main__":
    unittest.main()
