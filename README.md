# simplefci

## Introduction

A simple FCI (full configuration interaction) implementation in Python.
- Slater-Condon rule to evaluate the Hamiltonian matrix within the Slater 
determinant basis ( $ S_z $ eigen states), and diagonalize the Hamiltonian matrix
with the condense eigensolver in `scipy`.
- The “string-based” determinant-CI (or direct-CI) algorithm to generate the
matrix multiplication of the Hamiltonian matrix and use the sparse eigensolver
in `scipy` to diagonalize the Hamiltonian matrix.

In the tests, MOs and integrals are generated from `pyscf`.

## TODO:

- Davidson algorithm for matrix diagonalization and implement it for any 
symmetric matrix.

As the programs are completely written in Python, it should be able to solve 
problem with a maximum system size of ∼ 12 orbitals with 12 electrons.

## Acknowledgements

Thanks to Qiming Sun's tutorial and the guidance of Garnet Chan.
And the invaluable advice from Drs. Zhen Luo and Yihan Shao, and
the discussions with Rui Li, Bo Li and Zhihao Cui.

## References
[1] Szabo and Ostlund, _Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory_, Dover Publications, New York, 1996

[2] https://github.com/pyscf/pyscf