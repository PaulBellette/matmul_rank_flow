import mpmath as mp
from fractions import Fraction

from number_field_exact import SimpleNumberField, discover_common_field


def test_quadratic_and_biquadratic_recognition():
    mp.mp.dps = 100
    q = [(0, mp.sqrt(3)), (1, 1 + 2*mp.sqrt(3))]
    got = discover_common_field(q, max_degree=6, maxcoeff_algdep=10**15, maxcoeff_basis=10**18)
    assert got['diagnostics']['selected']['degree'] == 2

    b = [
        (0, mp.sqrt(2)),
        (1, mp.sqrt(3)),
        (2, 1 + mp.sqrt(2) + 2*mp.sqrt(3)),
        (3, mp.sqrt(6)),
    ]
    got = discover_common_field(b, max_degree=8, maxcoeff_algdep=10**18, maxcoeff_basis=10**20)
    assert got['diagnostics']['selected']['degree'] == 4
    assert set(got['representations']) == {0,1,2,3}


def test_cubic_and_field_arithmetic():
    mp.mp.dps = 100
    a = mp.root(2, 3)
    vals = [(0,a),(1,a*a+mp.mpf(1)/3),(2,2-a+4*a*a)]
    got = discover_common_field(vals, max_degree=6, maxcoeff_algdep=10**15, maxcoeff_basis=10**18)
    assert got['diagnostics']['selected']['degree'] == 3

    K = SimpleNumberField((Fraction(-2), Fraction(0), Fraction(0), Fraction(1)))
    alpha = K.elt([0,1,0])
    assert K.mul(K.mul(alpha, alpha), alpha) == K.from_rational(2)


def test_rank_over_number_field():
    K = SimpleNumberField((Fraction(-2), Fraction(0), Fraction(0), Fraction(1)))
    z, o = K.zero(), K.one()
    a = K.elt([0,1,0])
    M = [[o,z,z],[z,a,z],[z,z,o]]
    assert K.rank3(M) == 3
    M[2][2] = z
    assert K.rank3(M) == 2
