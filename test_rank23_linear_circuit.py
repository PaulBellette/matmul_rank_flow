from fractions import Fraction
from pathlib import Path

from number_field_exact import SimpleNumberField
from rank23_linear_circuit import analyze_certificate, greedy_linear_cse


def qfield():
    return SimpleNumberField((Fraction(0), Fraction(1)))


def e(field, x):
    return field.from_rational(Fraction(x))


def test_greedy_cse_reuses_shared_pair():
    f = qfield()
    z = e(f, 0)
    # x0+x1+x2 and x0+x1+x3: naive 4 additions, shared x0+x1 gives 3.
    forms = [
        (e(f,1), e(f,1), e(f,1), z),
        (e(f,1), e(f,1), z, e(f,1)),
    ]
    r = greedy_linear_cse(forms, f)
    assert r.naive_additions == 4
    assert r.greedy_cse_additions == 3


def test_existing_exact_seed_counts_match_complexity_campaign_starts():
    root = Path(__file__).resolve().parent
    r211 = analyze_certificate(root / "results/replication_5seeds/exact_endpoints/seed_211/exact/rank23_exact.json")
    r401 = analyze_certificate(root / "results/replication_5seeds/exact_endpoints/seed_401/exact/rank23_exact.json")
    assert r211["naive_additions"] == 143
    assert r401["naive_additions"] == 137
    assert r211["greedy_cse_additions"] <= 143
    assert r401["greedy_cse_additions"] <= 137
