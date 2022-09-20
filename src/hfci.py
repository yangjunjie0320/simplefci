import numpy

import scipy
from scipy.special import comb

# Note about naming convention:
# - occ means occupation number, which is 0 or 1
# - idx means index, which is 0, 1, 2, ...
# - diff means difference, which is 0, 1, -1
# - bin means a int64 element represents one occupations
#   in binary format.
# - configuration means a slater determinant, which contains
#   alpha and beta occupations.
# - anything works for one spin, for example `nelec_s`, `get_bins_s`
#   (get binary numbers to represent the configurations for one spin)
#   are marked with `..._s` suffix, or specific to alpha or beta spin,
# - anything works for a list, for example `nelecs`, `get_configs`
#   (get a list of configurations) 
#   are marked with plural suffix `...s`.
# - note that a tuple of two spin shall not be marked with any suffix,
#   for example `nelecs` is a tuple of two ints, which represents
#   except `nelecs` is a tuple of two ints.

class _Config(object):
    '''A class to represent a determinanat configuration.

    Attributes
    ----------
    occ_alph : list
        a list contain either 0 or 1 to present the 
        alpha configuration
    occ_beta : str
        a list contain either 0 or 1 to present the
        beta configuration
    nmo : int
        number of molecular orbitals
    nelecs : tuple[int, int]
        number of alpha and beta electrons
    '''

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

def get_configs(nmo=None, nelecs=None):
    '''Generate configurations for alpha and beta spins
    with given number of molecular orbitals.

    Parameters
    ----------
    nmo : int
        number of molecular orbitals
    nelecs : tuple[int, int]
        number of alpha and beta electrons

    Returns
    ----------
    configs : list[_Config]
        a list of _Config objects
    '''
    nelec_alph, nelec_beta = nelecs
    na = comb(nmo, nelec_alph, exact=True)
    nb = comb(nmo, nelec_beta, exact=True)

    bins_alph = get_bins_s(nmo, nelec_alph)
    bins_beta = get_bins_s(nmo, nelec_beta)

    assert len(bins_alph) == na
    assert len(bins_beta) == nb

    configs   = []

    for bin_alph in bins_alph:
        for bin_beta in bins_beta:
            occ_alph = [1 if bin_alph & (1 << p) else 0 for p in range(nmo)]
            occ_beta = [1 if bin_beta & (1 << p) else 0 for p in range(nmo)]
            configs.append(_Config(occ_alph, occ_beta, nmo=nmo, nelecs=nelecs))

    assert len(configs) == na * nb

    return configs

def get_bins_s(nmo, nelec_s):
    '''Generate binary numbers to represent the configurations
    for given spin (alpha or beta).

    Parameters
    ----------
    nmo : int
        number of molecular orbitals
    nelec_s : int
        number of electrons for given spin

    Returns
    ----------
    bins : list[int]
        - One int64 element represents one string in binary format.
        - Each bit stands for one the occupation number of one orbital.
        - The lowest (right-most) bit corresponds to the lowest orbital.
        - 1 means occupied and 0 means unoccupied.
    '''

    assert nelec_s >= 0 and nmo < 64

    if nelec_s == 0:
        return numpy.asarray([0], dtype=numpy.int64)

    elif nelec_s > nmo:
        return numpy.asarray([], dtype=numpy.int64)
        
    def _get_bins_s(nmo, nelec_s):
        '''A recursive function to generate binary numbers
        to represent the configurations.
        '''
        if nelec_s == 1:
            res = [(1 << i) for i in range(nmo)]

        elif nelec_s >= nmo:
            n = 0
            for i in range(nmo):
                n = n | (1 << i)
            res = [n]
            
        else:
            thisorb = 1 << (nmo - 1)
            res = _get_bins_s(nmo - 1, nelec_s)

            for n in _get_bins_s(nmo - 1, nelec_s - 1):
                res.append(n | thisorb)

        return res

    bins = _get_bins_s(nmo, nelec_s)
    return numpy.asarray(bins, dtype=numpy.int64)

class _ConfigDiff(object):
    '''A class to represent the difference between
    two configurations (for both spin).

    Attributes
    ----------
    occ_idx : tuple(list[int], list[int])
        a tuple of two lists of common occupied orbital 
        indices for each spin
    vir_idx : tuple(list[int], list[int])
        a tuple of two lists of common virtual orbital 
        indices for each spin
    diff_idx : tuple(list[int], list[int])
        ...

        a tuple of two lists of common virtual orbital 
        indices for each spin

        a tuple of two lists of orbital indices
        corresponding to the difference between
        two determinants
    '''

    def __init__(self, occ_idx, vir_idx, diff_idx):
        occ_idx_alph, occ_idx_beta = occ_idx
        vir_idx_alph, vir_idx_beta = vir_idx
        diff_idx_alph, diff_idx_beta = diff_idx

        self.occ_idx_alph = occ_idx_alph
        self.occ_idx_beta = occ_idx_beta
        self.vir_idx_alph = vir_idx_alph
        self.vir_idx_beta = vir_idx_beta
        self.diff_idx_alph = diff_idx_alph
        self.diff_idx_beta = diff_idx_beta
        
        diff_num = len(diff_idx[0])
        assert diff_num == len(diff_idx[0])
        assert diff_num == len(diff_idx[1])
        self.diff_num = diff_num

    def get_diff_idx(self):
        diff_num = self.diff_num
        diff_idx = self.diff_idx

        assert diff_num in [1, 2]

        if diff_num == 1:
            m = min(diff_idx[0][0], diff_idx[1][0])
            p = max(diff_idx[0][0], diff_idx[1][0])
            return m, p

        elif diff_num == 2:
            m, n = min(diff_idx[0]), max(diff_idx[0])   
            p, q = min(diff_idx[1]), max(diff_idx[1])

            return m, n, p, q

        else:
            return None

def get_occ_diff_s(occ1_s, occ2_s):
    '''Get the difference between two occupation lists
    for one spin.

    '''
    nmo = len(occ1_s)

    assert len(occ1_s) == nmo
    assert len(occ2_s) == nmo
    assert sum(occ1_s) == sum(occ2_s)

    comm_occ_idx_s = []
    comm_vir_idx_s = []

    diff_s = []

    for p in range(nmo):
        occ1_p_s = occ1_s[p]
        occ2_p_s = occ2_s[p]

        if occ1_p_s == occ2_p_s:
            if occ1_p_s == 0:
                comm_vir_idx_s.append(p)

            elif occ1_p_s == 1:
                comm_vir_idx_s.append(p)

            else:
                raise RuntimeError("Invalid Occupation Number")
        else:
            assert occ1_p_s in [0, 1]
            assert occ2_p_s in [0, 1]

        diff_s.append(occ1_p_s - occ2_p_s)

    diff_idx_1 = []
    diff_idx_2 = []

    for p in range(nmo):
        diff_p_s = diff_s[p]

        if diff_p_s == 1:
            diff_idx_1.append(p)
        elif diff_p_s == -1:
            diff_idx_2.append(p)
        else:
            assert p in comm_occ_idx_s or p in comm_vir_idx_s
            assert diff_p_s == 0

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

    occ_idx   = []
    vir_idx   = []
    diff_idx  = []

    for occ1_s, occ2_s in zip([occ1_alph, occ1_beta], [occ2_alph, occ2_beta]):
        occ_idx_s, vir_idx_s, diff_idx_s = get_occ_diff_s(occ1_s, occ2_s)

        occ_idx.append(occ_idx_s)
        vir_idx.append(vir_idx_s)
        diff_idx.append(diff_idx_s)

    diff_num  = 0

    for diff_idx_s in diff_idxs:
    if diff_idx_alph is not None:
        diff_num += len(diff_idx_alph[0])
    if diff_idx_beta is not None:
        diff_num += len(diff_idx_beta[0])

    config_diff = _ConfigDiff(occ_idx, vir_idx, diff_idx, diff_num)



    return config_diff

def get_fci_matrix_element(config1, config2, h1e, h2e, verbose = False):
    '''
    Returns the matrix element of the FCI Hamiltonian.
    '''

    nmo = h1e.shape[0]
    assert h1e.shape == (nmo, nmo)
    assert h2e.shape == (nmo, nmo, nmo, nmo)

    occ_idxs, vir_idxs, diff_idxs, diff_num = get_config_diff(config1, config2)
    diff_idx_alph, diff_idx_beta = diff_idxs
    occ_idx_alph, occ_idx_beta   = occ_idxs

    h = 0.0

    if diff_num == 0:
        for ia in occ_idx_alph:
            h += h1e[ia, ia]

            for ja in occ_idx_alph:
                h += 0.5 * h2e[ia, ia, ja, ja]
                h -= 0.5 * h2e[ia, ja, ja, ia]

            for jb in occ_idx_beta:
                h += 0.5 * h2e[ia, ia, jb, jb]
        
        for ib in occ_idx_beta:
            h += h1e[ib, ib]

            for ja in occ_idx_alph:
                h += 0.5 * h2e[ib, ib, ja, ja]

            for jb in occ_idx_beta:
                h += 0.5 * h2e[ib, ib, jb, jb]
                h -= 0.5 * h2e[ib, jb, jb, ib]

    elif diff_num == 1:
        if diff_idx_alph is not None:            
            assert diff_idx_beta is None
            assert diff_idx_alph[0][0] != diff_idx_alph[1][0]

            ma, pa = get_diff_idx(diff_idx_alph, diff_num)

            h += h1e[ma, pa]

            for ia in occ_idx_alph:
                h += h2e[ma, pa, ia, ia]
                h -= h2e[ma, ia, ia, pa]

            for ib in occ_idx_beta:
                h += h2e[ma, pa, ib, ib]

            h *= (- 1.0) ** sum(config1.occ_alph[ma+1:pa])

        elif diff_idx_beta is not None:
            assert diff_idx_alph is None
            assert diff_idx_beta[0][0] != diff_idx_beta[1][0]

            mb, pb = get_diff_idx(diff_idx_beta, diff_num)

            h += h1e[mb, pb]
            
            for ia in occ_idx_alph:
                h += h2e[mb, pb, ia, ia]

            for ib in occ_idx_beta:
                h += h2e[mb, pb, ib, ib]
                h -= h2e[mb, ib, ib, pb]
            
            h *= (- 1.0) ** sum(config1.occ_beta[mb+1:pb])
    
    elif diff_num == 2:

        if diff_idx_alph is None:

            mb, nb, pb, qb = get_diff_idx(diff_idx_beta, diff_num)

            h += h2e[mb, pb, nb, qb]
            h -= h2e[mb, qb, nb, pb]

            occ_beta = [1 if i in occ_idx_beta else 0 for i in range(nmo)]
            h *= (- 1.0) ** sum(occ_beta[min(mb,pb)+1:max(mb,pb)])
            h *= (- 1.0) ** sum(occ_beta[min(nb,qb)+1:max(nb,qb)])

        elif diff_idx_beta is None:
            ma, na, pa, qa = get_diff_idx(diff_idx_alph, diff_num)

            h += h2e[ma, pa, na, qa]
            h -= h2e[ma, qa, na, pa]

            occ_alph = [1 if i in occ_idx_alph else 0 for i in range(nmo)]
            h *= (- 1.0) ** sum(occ_alph[min(ma, pa)+1:max(ma, pa)])
            h *= (- 1.0) ** sum(occ_alph[min(na, qa)+1:max(na, qa)])

        else:
            ma, pa = get_diff_idx(diff_idx_alph, 1)
            nb, qb = get_diff_idx(diff_idx_beta, 1)

            h += h2e[ma, pa, nb, qb]

            h *= (- 1.0) ** sum(config1.occ_alph[ma+1:pa])
            h *= (- 1.0) ** sum(config1.occ_beta[nb+1:qb])

    return h

def get_hfci(h1e, h2e, nmo, nelecs):

    neleca, nelecb = nelecs
    
    na = get_num_configs(nmo, neleca)
    nb = get_num_configs(nmo, nelecb)

    bin_alph = make_occs(nmo, neleca)
    bin_beta = make_occs(nmo, nelecb)

    hfci = numpy.zeros((na * nb, na * nb))

    for ia in range(na):
        for ib in range(nb):
            for ja in range(na):
                for jb in range(nb):
                    bins_iaib = (bin_alph[ia], bin_beta[ib])
                    bins_jajb = (bin_alph[ja], bin_beta[jb])
                    config1 = get_config(bins_iaib, nmo, (neleca, nelecb))
                    config2 = get_config(bins_jajb, nmo, (neleca, nelecb))
                    hfci[ia * nb + ib, ja * nb + jb] = get_fci_matrix_element(config1, config2, h1e, h2e)

    return hfci