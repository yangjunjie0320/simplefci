from functools import reduce

import numpy
import scipy
from scipy import special

import pyscf
from pyscf import scf
from pyscf import ao2mo

def get_hamiltonian(mol: pyscf.gto.Mole):
    '''
    Returns the Hamiltonian in the MO basis.
    '''

    m = scf.RHF(mol)
    m.kernel()

    norb  = m.mo_coeff.shape[1]
    h1e   = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    h2e   = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
    h2e   = h2e.reshape(norb, norb, norb, norb)

    return h1e, h2e, norb, mol.nelec

def comb(nmo, nelec_s):
    return int(special.comb(nmo, nelec_s, exact=True))
    
