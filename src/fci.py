import numpy
import scipy

from hfci import get_hfci
from direct import get_hc_op

from utils import comb

def kernel(h1e, h2e, nmo, nelecs, nroots=1, tol=1e-8, method="slater-condon"):
    assert h1e.shape == (nmo, nmo)
    assert h2e.shape == (nmo, nmo, nmo, nmo)

    neleca, nelecb = nelecs
    na = comb(nmo, neleca)
    nb = comb(nmo, nelecb)

    if method == "slater-condon":
        hfci = get_hfci(h1e, h2e, nmo, nelecs)
        res  = scipy.linalg.eigh(hfci)

    elif method == "direct":
        hc_op = get_hc_op(h1e, h2e, nmo, nelecs)
        res   = scipy.sparse.linalg.eigsh(hc_op, k=nroots, tol=tol, which="SA")

    else:
        raise NotImplementedError("method = {}".format(method))

    efci = res[0][:nroots]
    if nroots == 1:
        ci   = res[1][:, 0].reshape(na, nb)
    else:
        ci   = res[1][:, :nroots].T.reshape(nroots, na, nb)

    return efci, ci
