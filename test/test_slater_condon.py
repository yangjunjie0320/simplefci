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
    efci, ci = kernel(h1e, h2e, nmo, nelecs, nroots=1, tol=TOL, method="slater-condon")

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
    efci, ci = kernel(h1e, h2e, nmo, nelecs, nroots=1, tol=TOL, method="slater-condon")

    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 100
    cisolver.conv_tol  = TOL
    cisolver.nroots    = 1
    efci0, ci0 = cisolver.kernel(h1e, h2e, nmo, nelecs)

    e_err = abs(efci - efci0)
    c_err = abs(1.0 - abs(numpy.einsum('ij,ij->', ci, ci0)))
    assert e_err < TOL
    assert c_err < TOL

def test_random():
    numpy.random.seed(12)

    for norb in range(2, 6):
        for neleca in range(1, norb + 1):
            for nelecb in range(1, norb + 1):

                nelecs = (neleca, nelecb)

                h1e = numpy.random.random((norb, norb))
                h2e = numpy.random.random((norb, norb, norb, norb))
                
                h1e = h1e + h1e.T
                h2e = h2e + h2e.transpose(1, 0, 2, 3)
                h2e = h2e + h2e.transpose(0, 1, 3, 2)
                h2e = h2e + h2e.transpose(2, 3, 0, 1)

                efci, ci = kernel(h1e, h2e, norb, nelecs, nroots=1, tol=TOL, method="slater-condon")

                cisolver = pyscf.fci.direct_spin1.FCI()
                cisolver.max_cycle = 100
                cisolver.conv_tol  = TOL
                cisolver.nroots    = 1
                efci0, ci0 = cisolver.kernel(h1e, h2e, norb, nelecs)

                e_err = abs(efci - efci0)
                c_err = abs(1.0 - abs(numpy.einsum('ij,ij->', ci, ci0)))

                assert e_err < TOL
                assert c_err < TOL

if __name__ == '__main__':
    test_h6()
    test_hf()
    test_random()
    