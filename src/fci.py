import sys
import numpy
import scipy
from functools import reduce

import pyscf
from pyscf import fci
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.fci import fci_slow
from pyscf.tools.dump_mat import dump_rec

from config import get_nconfig
from config import make_occs
from config import Config, get_config_diff

    
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

    cisolver = fci.direct_spin0.FCI()
    cisolver.max_cycle = 100
    cisolver.conv_tol  = 1e-8
    e, fcivec = cisolver.kernel(h1e, h2e, norb, mol.nelec)

    print(f"ehf = {m.energy_elec()[0]: 12.6f}, efci = {e: 12.6f}")

    return h1e, h2e, norb, mol.nelec

def get_diff_idx(diff_idx, diff_num):
    assert diff_num in [1, 2]

    if diff_num == 1:
        m = min(diff_idx[0][0], diff_idx[1][0])
        p = max(diff_idx[0][0], diff_idx[1][0])
        return m, p

    elif diff_num == 2:
        m, n = min(diff_idx[0]), max(diff_idx[0])   
        p, q = min(diff_idx[1]), max(diff_idx[1])

        return m, n, p, q

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
                # h -= h2e[ma, ib, ib, pa]

            h *= (- 1.0) ** sum(config1.occ_alph[ma+1:pa])

        elif diff_idx_beta is not None:
            assert diff_idx_alph is None
            assert diff_idx_beta[0][0] != diff_idx_beta[1][0]

            mb, pb = get_diff_idx(diff_idx_beta, diff_num)

            h += h1e[mb, pb]
            
            for ia in occ_idx_alph:
                h += h2e[mb, pb, ia, ia]
                # h -= h2e[mb, ia, ia, pb]

            for ib in occ_idx_beta:
                h += h2e[mb, pb, ib, ib]
                h -= h2e[mb, ib, ib, pb]
            
            h *= (- 1.0) ** sum(config1.occ_beta[mb+1:pb])
    
    elif diff_num == 2:
        if diff_idx_alph is None:
            if verbose:
                print("diff_num = 2")
                print(f"diff_idx_alph is None")

            mb, nb, pb, qb = get_diff_idx(diff_idx_beta, diff_num)

            h += h2e[mb, pb, nb, qb]
            h -= h2e[mb, qb, nb, pb]

            print(diff_idx_beta)
            print(f"mb = {mb}, nb = {nb}, pb = {pb}, qb = {qb}")
            print(f"h = {h}")
            h *= (- 1.0) ** sum(config1.occ_alph[min(mb,pb)+1:max(mb,pb)])
            h *= (- 1.0) ** sum(config1.occ_alph[min(nb,qb)+1:max(nb,qb)])
            print(f"h = {h}")
        
        elif diff_idx_beta is None:
            if verbose:
                print("diff_num = 2")
                print(f"diff_idx_beta is None")

            ma, na, pa, qa = get_diff_idx(diff_idx_alph, diff_num)

            h += h2e[ma, pa, na, qa]
            h -= h2e[ma, qa, na, pa]

            h *= (- 1.0) ** sum(config1.occ_alph[min(ma,pa)+1:max(ma,pa)])
            h *= (- 1.0) ** sum(config1.occ_alph[min(na,qa)+1:max(na,qa)])

        else:
            if verbose:
                print("diff_num = 2")
                print(f"diff_idx_alph is not None")
                print(f"diff_idx_beta is not None")
                assert 1 == 2
                
            ma, pa = get_diff_idx(diff_idx_alph, 1)
            nb, qb = get_diff_idx(diff_idx_beta, 1)

            h += h2e[ma, pa, nb, qb]

            h *= (- 1.0) ** sum(config1.occ_alph[ma+1:pa])
            h *= (- 1.0) ** sum(config1.occ_beta[nb+1:qb])

    return h

def get_hfci(h1e, h2e, nmo, nelecs, check = True, stdout = sys.stdout):

    neleca, nelecb = nelecs
    
    na = fci.cistring.num_strings(nmo, neleca)
    nb = fci.cistring.num_strings(nmo, nelecb)

    occs_alph = make_occs(nmo, neleca)
    occs_beta = make_occs(nmo, nelecb)

    hfci = numpy.zeros((na * nb, na * nb))

    if check:
        ci = numpy.eye(na * nb)
        h  = fci_slow.absorb_h1e(h1e, h2e, nmo, nelecs, .5)
                
        def hop(c):
            hc = fci_slow.contract_2e(h, c, nmo, nelecs)
            return hc.reshape(-1)

        hfci_ref = []

        for ii in range(na * nb):
            h_ci = hop(ci[ii].reshape(na, nb))
            hfci_ref.append(h_ci)

        hfci_ref = numpy.asarray(hfci_ref).reshape(na * nb, na * nb)

        for ia in range(na):
            for ib in range(nb):
                for ja in range(na):
                    for jb in range(nb):
                        config1 = Config(occs_alph[ia], occs_beta[ib], nmo, (neleca, nelecb))
                        config2 = Config(occs_alph[ja], occs_beta[jb], nmo, (neleca, nelecb))

                        hfci[ia * nb + ib, ja * nb + jb] = get_fci_matrix_element(config1, config2, h1e, h2e)

                        if abs(hfci[ia * nb + ib, ja * nb + jb] - hfci_ref[ia * nb + ib, ja * nb + jb]) > 1e-6:
                            occ_idxs, vir_idxs, diff_idxs, diff_num = get_config_diff(config1, config2)
                            print()
                            assert diff_num == 2
                            print("Error:", hfci[ia * nb + ib, ja * nb + jb], hfci_ref[ia * nb + ib, ja * nb + jb])
                            print(config1.occ_alph, config1.occ_beta)
                            print(config2.occ_alph, config2.occ_beta)
                            
                            print(f"{hfci[ia * nb + ib, ja * nb + jb] :12.6f}")
                            print(f"{hfci_ref[ia * nb + ib, ja * nb + jb] :12.6f}")

                            get_fci_matrix_element(config1, config2, h1e, h2e, verbose = True)

        
        efci       = scipy.linalg.eigh(hfci)[0][0]
        efci_ref_0 = fci_slow.kernel(h1e, h2e, nmo, nelecs[0] + nelecs[1])
        efci_ref_1 = scipy.linalg.eigh(hfci_ref)[0][0]

        print(f"{efci :12.6f}", f"{efci_ref_0 :12.6f}", f"{efci_ref_1 :12.6f}")

        assert abs(efci_ref_0 - efci_ref_1) < 1e-6
        assert abs(efci_ref_0 - efci)       < 1e-6
        assert abs(efci_ref_1 - efci)       < 1e-6
    
    else:

        for ia in range(na):
            for ib in range(nb):
                for ja in range(na):
                    for jb in range(nb):
                        config1 = Config(occs_alph[ia], occs_beta[ib], nmo, (neleca, nelecb))
                        config2 = Config(occs_alph[ja], occs_beta[jb], nmo, (neleca, nelecb))

                        hfci[ia * nb + ib, ja * nb + jb] = get_fci_matrix_element(config1, config2, h1e, h2e)


        

    return hfci
        

def kernel(h1e, h2e, nmo, nelecs):
    assert h1e.shape == (nmo, nmo)
    assert h2e.shape == (nmo, nmo, nmo, nmo)

    h = get_hfci(h1e, h2e, nmo, nelecs, check = True)

    
