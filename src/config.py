import numpy
import scipy
from scipy.special import comb
from functools import reduce

import pyscf

class _Config(object):
    """
    A class to represent a determinanat configuration.

    Attributes
    ----------
    occ_alph : str
        a formatted string to print out what the animal says
    occ_beta : str
        the name of the animal
    nmo : str
        the sound that the animal makes
    nelecs : int
        the number of legs the animal has (default 4)
    """

    def __init__(self, occ_alph, occ_beta, nmo=None, nelecs=None):
        assert nmo == len(occ_alph)
        assert nmo == len(occ_beta)

        neleca, nelecb = nelecs
        assert neleca == sum(occ_alph)
        assert nelecb == sum(occ_beta)

        self.nmo    = nmo
        self.nelecs = nelecs

        self.occ_alph = occ_alph
        self.occ_beta = occ_beta

def get_config(occ_alph, occ_beta, nmo=None, nelecs=None):
    if nmo is None:
        nmo = len(occ_alph)

    if nelecs is None:
        nelecs = (sum(occ_alph), sum(occ_beta))

    return _Config(occ_alph, occ_beta, nmo, nelecs)

class _ConfigDiff(object):
    pass

def get_nconfig(nmo, nelec):
    return comb(nmo, nelec, exact=True)

def make_bins(nmo, nelec):
    '''Generate string from the given orbital list.

    Returns:
        list of int64.  One int64 element represents one string in binary format.
        The binary format takes the convention that the one bit stands for one
        orbital, bit-1 means occupied and bit-0 means unoccupied.  The lowest
        (right-most) bit corresponds to the lowest orbital in the orb_list.

    '''

    assert nmo < 64

    assert(nelec >= 0)
    if nelec == 0:
        return numpy.asarray([0], dtype=numpy.int64)

    elif nelec > nmo:
        return numpy.asarray([], dtype=numpy.int64)
        
    def gen_str_iter(nmo, nelec):
        if nelec == 1:
            res = [(1 << i) for i in range(nmo)]

        elif nelec >= nmo:
            n = 0
            for i in range(nmo):
                n = n | (1 << i)
            res = [n]
            
        else:
            thisorb = 1 << (nmo - 1)
            res = gen_str_iter(nmo - 1, nelec)

            for n in gen_str_iter(nmo - 1, nelec - 1):
                res.append(n | thisorb)

        return res

    bins = gen_str_iter(nmo, nelec)
    assert len(bins) == get_nconfig(nmo, nelec)

    return numpy.asarray(bins, dtype=numpy.int64)

def _occ_from_bin(ci_bin, nmo):
    occ = []

    ci_string = bin(ci_bin)[2:][::-1]

    for p in range(nmo):
        if p < len(ci_string):
            occ.append(1 if ci_string[p] == '1' else 0)

        else:
            occ.append(0)
    
    return occ

def make_occs(nmo, nelec):
    bins = make_bins(nmo, nelec)
    return [_occ_from_bin(b, nmo) for b in bins]

def get_occ_diff(occ1, occ2):
    nmo = len(occ1)

    assert len(occ1) == nmo
    assert len(occ2) == nmo
    assert sum(occ1) == sum(occ2)

    occ_idx    = []
    vir_idx    = []
    diff_idx_1 = []
    diff_idx_2 = []

    diff     = []

    for p in range(nmo):
        occ1_p = occ1[p]
        occ2_p = occ2[p]

        if occ1_p == occ2_p:
            if  occ1_p == 0:
                vir_idx.append(p)

            elif occ1_p == 1:
                occ_idx.append(p)

            else:
                raise RuntimeError("Invalid Occupation Number")
        else:
            assert occ1_p in [0, 1]
            assert occ2_p in [0, 1]

        diff.append(occ1_p - occ2_p)

    for p in range(nmo):
        diff_p = diff[p]

        if diff_p == 1:
            diff_idx_1.append(p)
        elif diff_p == -1:
            diff_idx_2.append(p)
        else:
            assert diff_p == 0

    assert len(diff_idx_1) == len(diff_idx_2)

    diff_idx = None

    if len(diff_idx_1) == 0:
        diff_idx = None
    else:
        diff_idx = (diff_idx_1, diff_idx_2)

    return occ_idx, vir_idx, diff_idx

def get_config_diff(config1, config2):
    occ1_alph = config1.occ_alph
    occ1_beta = config1.occ_beta
    occ2_alph = config2.occ_alph
    occ2_beta = config2.occ_beta

    occ_idx_alph, vir_idx_alph, diff_idx_alph = get_occ_diff(occ1_alph, occ2_alph)
    occ_idx_beta, vir_idx_beta, diff_idx_beta = get_occ_diff(occ1_beta, occ2_beta)

    occ_idxs  = (occ_idx_alph, occ_idx_beta)
    vir_idxs  = (vir_idx_alph, vir_idx_beta)
    diff_idxs = (diff_idx_alph, diff_idx_beta)

    diff_num  = 0
    if diff_idx_alph is not None:
        diff_num += len(diff_idx_alph[0])
    if diff_idx_beta is not None:
        diff_num += len(diff_idx_beta[0])

    return occ_idxs, vir_idxs, diff_idxs, diff_num
