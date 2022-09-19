from functools import reduce

import numpy
import scipy

import pyscf
from pyscf import fci
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.fci import fci_slow
from pyscf.tools.dump_mat import dump_rec

def get_hamiltonian(mol: pyscf.gto.Mole):
    '''
    Returns the Hamiltonian in the MO basis.
    '''

    m = scf.RHF(mol)
    m.kernel()

    norb   = m.mo_coeff.shape[1]
    h1e   = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    h2e   = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
    h2e   = h2e.reshape(norb, norb, norb, norb)

    return h1e, h2e, norb, mol.nelec