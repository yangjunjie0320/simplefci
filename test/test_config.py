from hfci import get_occ_diff
from hfci import make_bins
from hfci import make_occs
from hfci import _occ_from_bin

def test_get_occ_diff():

    occ1 = [1, 1, 1, 1, 0, 0]
    occ2 = [1, 1, 1, 1, 0, 0]

    res      = get_occ_diff(occ1, occ2)
    occ_idx  = res[0]
    vir_idx  = res[1]
    diff_idx = res[2]

    assert occ_idx  == [0, 1, 2, 3]
    assert vir_idx  == [4, 5]
    assert diff_idx == None

    occ1 = [1, 1, 1, 1, 0, 0]
    occ2 = [1, 1, 1, 0, 1, 0]

    res      = get_occ_diff(occ1, occ2)
    occ_idx  = res[0]
    vir_idx  = res[1]
    diff_idx = res[2]

    assert occ_idx  == [0, 1, 2]
    assert vir_idx  == [5]
    assert diff_idx == ([3], [4])

    occ1 = [1, 1, 1, 1, 0, 0]
    occ2 = [1, 1, 0, 0, 1, 1]

    res      = get_occ_diff(occ1, occ2)
    occ_idx  = res[0]
    vir_idx  = res[1]
    diff_idx = res[2]

    assert occ_idx  == [0, 1]
    assert vir_idx  == []
    assert diff_idx == ([2, 3], [4, 5])

def test_make_bins():
    ref = ['0b11', '0b101', '0b110', '0b1001', '0b1010', '0b1100',
            '0b10001', '0b10010', '0b10100', '0b11000']

    for i, x in enumerate(make_bins(5, 2)):
        assert bin(x) == ref[i]
        xx = _occ_from_bin(x, 5)

    for i, x in enumerate(make_occs(5, 2)):
        print(x)

if __name__ == "__main__":
    test_get_occ_diff()
    test_make_bins()