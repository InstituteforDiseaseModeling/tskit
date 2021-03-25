#! /usr/bin/env python3

from pathlib import Path
import tskit

sequence_file = Path("/Users/christopher/Coding/PycharmProjects/tskit_explore/tree-sequence.ts")
sequence = tskit.TreeSequence.load(sequence_file)
sequence
genomes = sequence.genomes()
print(f"{genomes.shape=}")
print(f"{genomes[0,:]=}")
print(f"{genomes[-1,:]=}")
print(f"{tskit.calculate_ibx(genomes, None, None)=}")

"""
python3
import tskit
seq = tskit.TreeSequence.load("tree-sequence.ts")
gen = seq.genomes()
ibx = tskit.calculate_ibx(gen)
print("\n")
from hashlib import sha256
for i in range(gen.shape[0]):
    print(f"{i}: {sha256(gen[i,:]).hexdigest()}")

"""

"""
 0/ 0: 07707846d42ad8a9f058f197f570ca0f762bb47f7bb27bf97532604c0d376517
 1/ 1: a90356b62320c2b263059b3fbe6e5eb3bea394fba1233b3dec03b4cd7cd0b9c7
 2/ 2: a89f6478a8be9a438ddde44d779a2029b8fd428f9b5921f8a653ce8cf7d46a9d
 3/ 3: 07707846d42ad8a9f058f197f570ca0f762bb47f7bb27bf97532604c0d376517
 4/ 4: 728828b4d1849ba073ba8d55d5dec3e520a80afe22cb4b7b759595e542a3fa58
 5/ 5: b8e7eb2d48774c4e7d40108d9f6c7a3d2722dd158722a87d3cc902e4ca56939a
 6/ 6: 0570bc05dab8a2fa1c1d2b993c97bee439f737cc122abbab0b1ca3fa63ed14d8
 7/ 7: ef14df001bcac8357a668ed538d4a3777f57485dce358a111d963b5ad4b05fa3
 8/ 8: dd19805b48cd9b3c1af0655b27157a3dcf840baceb530cedc402df219d7b3e64
 9/ 9: bc885256bea29b6c07ae9f91b2b851ad700a8920282a87e8663ec750cf54e0d0
10/10: 5eebe71cd9ada46b68f0046329a83ca4e91a8e3a888aeed0a50f98efeadb9994
11/11: 07707846d42ad8a9f058f197f570ca0f762bb47f7bb27bf97532604c0d376517
12/12: 2bde0bcbb4ddfe74eb3b17064009a73916c91f7476dae1d54af8a6dcc0be9857

 0: 07707846d42ad8a9f058f197f570ca0f762bb47f7bb27bf97532604c0d376517 - check
 1: a90356b62320c2b263059b3fbe6e5eb3bea394fba1233b3dec03b4cd7cd0b9c7
 2: a89f6478a8be9a438ddde44d779a2029b8fd428f9b5921f8a653ce8cf7d46a9d
 3: 07707846d42ad8a9f058f197f570ca0f762bb47f7bb27bf97532604c0d376517
 4: 728828b4d1849ba073ba8d55d5dec3e520a80afe22cb4b7b759595e542a3fa58
 5: b8e7eb2d48774c4e7d40108d9f6c7a3d2722dd158722a87d3cc902e4ca56939a
 6: 0570bc05dab8a2fa1c1d2b993c97bee439f737cc122abbab0b1ca3fa63ed14d8
 7: ef14df001bcac8357a668ed538d4a3777f57485dce358a111d963b5ad4b05fa3
 8: dd19805b48cd9b3c1af0655b27157a3dcf840baceb530cedc402df219d7b3e64
 9: bc885256bea29b6c07ae9f91b2b851ad700a8920282a87e8663ec750cf54e0d0
10: 5eebe71cd9ada46b68f0046329a83ca4e91a8e3a888aeed0a50f98efeadb9994
11: 07707846d42ad8a9f058f197f570ca0f762bb47f7bb27bf97532604c0d376517
12: 2bde0bcbb4ddfe74eb3b17064009a73916c91f7476dae1d54af8a6dcc0be9857
"""

"""
import _idm
import numpy as np
gen = np.random.randint(0, 8, (8192, 131072), dtype=np.uint16)
ids = np.asarray(range(16), dtype=np.uint32)
ibx = _idm.calculate_ibx(gen, ids)
"""
