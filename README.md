# simplefci

## Introduction

A simple FCI (full configuration interaction) implementation in Python.
Use the Slater-Condon rule to evaluate the Hamiltonian within the Slater 
determinant basis ($S_z$ eigen states), with the MOs and integrals generated
from `pyscf`; then directly diagonalize it using `numpy`.

## TODO:

- A faster and more technical implementation, the so-called “string-based” 
determinant-CI (or direct-CI) algorithm.
- Davidson algorithm for matrix diagonalization and implement it for any 
symmetric matrix.
- Knowles and Handy’s paper (1984).
- Implement functions to handle the FCI string, then 
the $\mathbf{H} \mathbf{C}$ operation.
- Put everything together. Davidson diagonalization solver;
preconditioner (can be taken from your program in the previsou step);
$\mathbf{H} \mathbf{C}$ operation, debug and tests.

If the programs are completely written in Python, it should be able to solve 
problem with a maximum system size of ∼ 12 orbitals with 12 electrons.

## Acknowledgements

Thanks to Qiming Sun's tutorial and the guidance of Garnet Chan.
And the invaluable advice from Drs. Zhen Luo and Yihan Shao, and
the discussions with Rui Li, Bo Li and Zhihao Cui.

## References
[1] Szabo and Ostlund, _Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory_, Dover Publications, New York, 1996

[2] https://github.com/pyscf/pyscf