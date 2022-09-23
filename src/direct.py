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

    return numpy.einsum('pq,pqJI->JI', h1e, t1, optimize=True)

def _contract_h2e(h2e, c, norb, nelecs):
    na, nb         = c.shape
    neleca, nelecb = nelecs

    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = c.reshape(na,nb)
    t1 = numpy.zeros((norb,norb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * ci0[:,str0]

    from pyscf import lib
    t1 = lib.einsum('bjai,aiAB->bjAB', h2e.reshape([norb]*4), t1)

    fcinew = numpy.zeros_like(ci0)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1[a,i,str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * t1[a,i,:,str0]

    return fcinew.reshape(c.shape)

def get_hc_op(h1e, h2e, nmo, nelecs):
    assert h1e.shape == (nmo, nmo)
    assert h2e.shape == (nmo, nmo, nmo, nmo)

    neleca, nelecb = nelecs
    na = comb(nmo, neleca)
    nb = comb(nmo, nelecb)

    g2e = 0.5 * h2e
    for k in range(nmo):
        g2e[k,k,:,:] -= numpy.einsum('jiik->jk', h2e) * (0.25 / (neleca + nelecb + 1e-100))
        g2e[:,:,k,k] -= numpy.einsum('jiik->jk', h2e) * (0.25 / (neleca + nelecb + 1e-100))

    def hh(v):
        hv  = _contract_h1e(h1e, v.reshape(na, nb), nmo, nelecs)
        hv += _contract_h2e(g2e, v.reshape(na, nb), nmo, nelecs)
        
        return hv.reshape(-1)

    hc_op = LinearOperator((na*nb, na*nb), matvec=hh)

    return hc_op