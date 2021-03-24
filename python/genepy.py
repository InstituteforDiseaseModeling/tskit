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
