# IDM Extensions to TSKIT

## Getting Set Up Locally

**Note:** TSKIT now requires Python 3.7 or newer (does not work with Python 3.6)

1. `git clone https://github.com/clorton/tskit.git`<br>_OR_<br>`git clone git@github.com:clorton/tskit.git`
2. in the repository, `git submodule init` (because TSKIT includes KASTORE as a submodule)
3. in the repository, `git submodule update` (see above)
4. (recommended) `git remote rename origin clorton` (we will use "clorton" as the remote name in the following command)
5. `git checkout -b idm-ibx-calculations clorton/idm-ibx-calculations`
6. <font color=orange>**change to the python folder in the repository**</font>

### To Build for a Local _Install_ and Install

7. `python3 setup.py bdist_wheel`
8. `python3 -m pip install dist/<name of wheel>` (\<_name of wheel_\> depends on your Python version)

### To Build for _Development_ and Install

7. `python3 -m pip install -e .`

## Basic Usage

With a TSKIT TreeSequence (e.g. `ts = tskit.load("tree-sequence.ts"))`)...

0. `import idm`
1. get the "full genome" data for the genomes in the tree:

```python
genomes, lengths = idm.get_genomes(ts)
```

### For IB**D** Data

2. `ibd = idm.IbxResults(genomes, intervals=lengths)`
3. look up IBD for a pair of genomes, g<sub>x</sub> and g<sub>y</sub>, with `similarity = ibd[gx, gy]`

The value returned from the IbxResults lookup is the sum of the interval lengths of the intervals which the two genomes inherit from the same root. Divide this value by the total length of all the intervals* to normalize to [0.0 and 1.0].

\*e.g., `np.sum(lengths)`

Optionally, you can request the IBD for only a subset of the available genomes be calculated and returned (can save time and memory).

Asumming you have `import numpy as np` and a list of genome IDs, `ids`:

* `subset = np.asarray(ids, dtype=np.uint32)`
* `ibd = idm.IbxResults(genomes, intervals=lengths, indices=subset)`

### For IB**S** Data

Omit the `intervals` parameter to `idm.IbxResults()` and the calculation will assume each position in the genome is of length 1. You can still, optionally, provide a subset of the available genome IDs, if appropriate, to reduce both computation time and memory usage.

The value returned from the IbxResults lookup is the number of sites with matching variants. Divide this value by the total number of sites to normalize to [0.0...1.0].

## Caveats and Warnings

## "Low Level" API

* `align_data(ndarray)`  
<br>Takes a two dimensional NumPy array and returns an equivalent array which is allocated and strided so that every row of the array, e.g. array[N,:] starts on a 32-byte boundary.

* `get_genomes(treesequence)`  
<br>Takes a TSKIT TreeSequence of G genomes and I intervals and returns a two-tuple of a two dimensional NumPy array with the "full genome" data, i.e. the source root of each interval in each genome, with shape (G,I) and a one dimensional NumPy array with the intervals lengths for all I intervals.  
<br>The genome data is memory aligned to be used directly in `calculate_ibx()`.

* `calculate_ibx(genomes:ndarray, intervals:ndarray=None, indices:ndarray=None)`  
<br>Takes a two dimensional NumPy array of genome data, e.g. from `get_genomes()`, and returns an array with the similarity between pairs of genomes as calculated with IBD or IBS.  
Optionally takes `intervals` and/or `indices` parameters.  
<br>The `intervals` parameter, if passed, should be a one-dimensional NumPy array, `dtype=np.uint32`, with the lengths of the intervals in the genomes. `intervals.shape[0] == genomes.shape[1]`  
<br>The `indices` parameter, if passed, should be a one-dimensional NumPy array, `dtype=np.uint32`, with genome IDs to be considered in the calculation. Entries in `indices` should be in the range [0...G).  
<br>Returns a three tuple (thruple?) of a two dimensional NumPy array with similarity data, a list of SHA256 hash digests, and a dictionary mapping hash digests to rows/columns in the similarity data.

### Gory Details

If there happen to be a significant number of clonal copies of a genome in the TreeSequence, we can save quite a bit of time by only calculating the similarity between unique genomes. E.g., if there are, due to clonal copies, only G/2 unique genomes in the set, we can get a 4x speed up in calculation. We determine which genomes are clonal copies by calculating the SHA256 hash of each genome's data (much faster than comparing the full genomic data of the two genomes). We then only calculate the similarity between unique genomes in this reduced set of comparisons. To determine the relevant row/column in the reduced array for a given genome, g<sub>i</sub>, use the following procedure:

First, determine the SHA256 hash of g<sub>i</sub> from the list of hash digests returned from `calculate_ibx()`, e.g.:

```python
hash = hashes[gi]
```

Next, determine the index of that hash digest in the sorted, unique set of hash digests by looking it up in the mapping returned from `calculate_ibx()`, e.g. `index = mapping[hash]`

Last, look up the similarity value for the genome g<sub>i</sub> and another genome, g<sub>j</sub> in the reduced array returned from `calculate_ibx()`, e.g.:

```python
similarity = ibx[index_for_gi, indexfor_gj]
```

This can be put together for the following:

```python
similarity = ibx[mapping[hashes[i]], mapping[hashes[j]]]
```

Note that the `IbxResults` class captures the hash list and mapping returned from `calculate_ibx()` and does this look up for you so you can index the results in an instance of `IbxResults` directly with the original genome IDs: `similarity = results[i, j]`
