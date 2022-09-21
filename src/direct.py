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

    h1e_c = numpy.dot(h1e.reshape(-1), t1.reshape(-1,na*nb))

    return h1e_c.reshape(na, nb)


def _contract_h2e(h2e, c, norb, nelecs):
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

    t1 = numpy.einsum('pqrs,rsIJ->pqIJ', h2e, t1, optimize=True)

    h2e_c = numpy.zeros_like(c)
    for str0, tab in enumerate(link_index_alph):
        for a, i, str1, sign in tab:
            h2e_c[str1] += sign * t1[a,i,str0]

    for str0, tab in enumerate(link_index_beta):
        for a, i, str1, sign in tab:
            h2e_c[:,str1] += sign * t1[a,i,:,str0]

    return h2e_c.reshape(na, nb)

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    from pyscf import ao2mo
    if not isinstance(nelec, (int, numpy.integer)):
        nelec = sum(nelec)
    h2e = ao2mo.restore(1, eri.copy(), norb)
    f1e = h1e - numpy.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return h2e * fac

def get_hc_op(h1e, h2e, nmo, nelecs):
    assert h1e.shape == (nmo, nmo)
    assert h2e.shape == (nmo, nmo, nmo, nmo)

    neleca, nelecb = nelecs
    na = comb(nmo, neleca)
    nb = comb(nmo, nelecb)

    h = absorb_h1e(h1e, h2e, nmo, nelecs, fac=1)

    def hh(v):
        # hc  = _contract_h1e(h1e, v.reshape(na, nb), nmo, nelecs)
        hc = _contract_h2e(h, v.reshape(na, nb), nmo, nelecs)
        return hc.reshape(-1)/2

    v0 = numpy.zeros((na*nb, ))
    v0[0] = 1.0

    hc_op = LinearOperator((na*nb, na*nb), matvec=hh)

    return hc_op