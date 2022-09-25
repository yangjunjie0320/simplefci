import numpy
import scipy

import pyscf
from pyscf import fci

from fci import kernel
from utils import get_hamiltonian

TOL = 1e-6

def test_h6():
    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis  = 'sto3g'
    mol.build()

    h1e, h2e, nmo, nelecs = get_hamiltonian(mol)
    efci, ci = kernel(h1e, h2e, nmo, nelecs, nroots=1, tol=TOL, method="direct")

    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 100
    cisolver.conv_tol  = TOL
    cisolver.nroots    = 1
    efci0, ci0 = cisolver.kernel(h1e, h2e, nmo, nelecs)

    e_err = abs(efci - efci0)
    c_err = abs(1.0 - abs(numpy.einsum('ij,ij->', ci, ci0)))
    assert e_err < TOL
    assert c_err < TOL

def test_hf():
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'sto3g',
        symmetry = False,
        verbose = 0,
    )

    h1e, h2e, nmo, nelecs = get_hamiltonian(mol)
    efci, ci = kernel(h1e, h2e, nmo, nelecs, nroots=1, tol=TOL, method="direct")

    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 100
    cisolver.conv_tol  = TOL
    cisolver.nroots    = 1
    efci0, ci0 = cisolver.kernel(h1e, h2e, nmo, nelecs)

    e_err = abs(efci - efci0)
    c_err = abs(1.0 - abs(numpy.einsum('ij,ij->', ci, ci0)))
    assert e_err < TOL
    assert c_err < TOL

if __name__ == '__main__':
    test_h6()
    test_hf()
