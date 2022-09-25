import numpy
import scipy
from scipy.sparse.linalg import LinearOperator

from hfci import get_hfci
from utils import comb

from pyscf.fci import cistring
from pyscf.fci import fci_slow

def _contract_h1e(h1e, c, norb, nelecs):
    na, nb         = c.shape
    neleca, nelecb = nelecs

    t1  = numpy.zeros((norb, norb, na, nb))
    link_index_alph = cistring.gen_linkstr_index(range(norb), neleca)
    link_index_beta = cistring.gen_linkstr_index(range(norb), nelecb)

    for str0, tab in enumerate(link_index_alph):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * c[str0]

    for str0, tab in enumerate(link_index_beta):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * c[:,str0]

    h1e_c = numpy.einsum('pq,pqJI->JI', h1e, t1, optimize=True)
    return h1e_c

def _contract_h2e(g2e, c, norb, nelecs):
    na, nb         = c.shape
    neleca, nelecb = nelecs

    t1  = numpy.zeros((norb, norb, na, nb))
    link_index_alph = cistring.gen_linkstr_index(range(norb), neleca)
    link_index_beta = cistring.gen_linkstr_index(range(norb), nelecb)

    for str0, tab in enumerate(link_index_alph):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * c[str0]

    for str0, tab in enumerate(link_index_beta):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * c[:,str0]

    t1 = numpy.einsum('bjai,aiAB->bjAB', g2e, t1, optimize=True)

    h2e_c = numpy.zeros_like(c)
    for str0, tab in enumerate(link_index_alph):
        for a, i, str1, sign in tab:
            h2e_c[str1] += sign * t1[a,i,str0]

    for str0, tab in enumerate(link_index_beta):
        for a, i, str1, sign in tab:
            h2e_c[:,str1] += sign * t1[a,i,:,str0]

    return h2e_c

def get_hc_op(h1e, h2e, nmo, nelecs):
    '''Generate the linear operator for the FCI matrix vector product.
    Will be passed to scipy.sparse.linalg.eigsh as the sparse matrix
    multiplication operator.

    Parameters
    ----------
    h1e : numpy.ndarray
        One-electron Hamiltonian.
    h2e : numpy.ndarray
        Two-electron Hamiltonian.
    nmo : int
        Number of molecular orbitals.
    nelecs : tuple
        Number of alpha and beta electrons.

    Returns
    ----------
    hc_op : scipy.sparse.linalg.LinearOperator
        Linear operator for the FCI matrix vector product.
    '''
    assert h1e.shape == (nmo, nmo)
    assert h2e.shape == (nmo, nmo, nmo, nmo)

    neleca, nelecb = nelecs
    na = comb(nmo, neleca)
    nb = comb(nmo, nelecb)

    g2e = 0.5 * h2e
    for k in range(nmo):
        g2e[k,k,:,:] -= numpy.einsum('jiik->jk', h2e) * (0.25 / (neleca + nelecb + 1e-100))
        g2e[:,:,k,k] -= numpy.einsum('jiik->jk', h2e) * (0.25 / (neleca + nelecb + 1e-100))

    def matvec(v):
        hv  = _contract_h1e(h1e, v.reshape(na, nb), nmo, nelecs)
        hv += _contract_h2e(g2e, v.reshape(na, nb), nmo, nelecs)
        
        return hv.reshape(-1)

    hc_op = LinearOperator((na*nb, na*nb), matvec=matvec)

    return hc_op

