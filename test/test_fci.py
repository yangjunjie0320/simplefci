import numpy
import scipy

import pyscf

from fci import kernel
from fci import get_hamiltonian

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
print(nmo, nelecs)

kernel(h1e, h2e, nmo, nelecs)