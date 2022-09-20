import numpy

import scipy
from scipy.special import comb

from hfci import get_hfci
        

def kernel(h1e, h2e, nmo, nelecs, method="slater-condon"):
    assert h1e.shape == (nmo, nmo)
    assert h2e.shape == (nmo, nmo, nmo, nmo)

    neleca, nelecb = nelecs
    na = comb(nmo, neleca, exact=True)
    nb = comb(nmo, nelecb, exact=True)

    efci = None
    ci   = None

    if method == "slater-condon":
        hfci = get_hfci(h1e, h2e, nmo, nelecs, check = True)
        res  = scipy.linalg.eigh(hfci)
        efci = res[0][0]
        ci   = res[1][:, 0].reshape(na, nb)

    else:
        raise NotImplementedError

    return efci, ci
