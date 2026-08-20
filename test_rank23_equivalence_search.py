from pathlib import Path
import random
import sympy as sp

from rank23_equivalence_search import (
    Scheme, canonical_wl_fingerprint, load_scheme_json, load_scheme_exp,
    _split_exp_product, rank_summary, search, projective_sandwich_test,
)

ROOT = Path(__file__).parent
CERT = ROOT / 'results/blind_rank23/exact/rank23_exact.json'


def _simple_gl():
    # Well-conditioned monomial/rational basis changes keep the symbolic test fast
    # while still exercising the full cyclic sandwich convention.
    P = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    Q = sp.diag(2, -1, 1)
    R = sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return P, Q, R


def _transform(s, seed=1):
    P,Q,R = _simple_gl()
    A,B,C=s.factors
    Ai=[P*M*Q.inv() for M in A]
    Bi=[Q*M*R.inv() for M in B]
    Ci=[R*M*P.inv() for M in C]
    perm=list(range(s.rank)); random.Random(seed+9).shuffle(perm)
    t = Scheme('transformed', ([Ai[i] for i in perm],[Bi[i] for i in perm],[Ci[i] for i in perm]), 'memory')
    # source channel r lives at target index invperm[r]
    invperm = {old: new for new, old in enumerate(perm)}
    return t, invperm


def test_exact_certificate_invariants():
    s=load_scheme_json(CERT); assert s is not None
    q=rank_summary(s)
    assert tuple(sorted(q['leg_rank_sums'])) == (27,32,33)
    assert q['g'] == 'w^33 + w^32 + w^27'
    assert sum(v for _,v in q['triple_counts']) == 23
    assert q['factor_counts'] == (0,46,23,0)


def test_wl_invariant_under_sandwich_and_channel_permutation():
    s=load_scheme_json(CERT); assert s is not None
    t,_=_transform(s)
    assert rank_summary(s)['canonical_pattern'] == rank_summary(t)['canonical_pattern']
    assert canonical_wl_fingerprint(s)[0] == canonical_wl_fingerprint(t)[0]


def test_projective_sandwich_recovers_known_transform():
    s=load_scheme_json(CERT); assert s is not None
    t,mapping=_transform(s, 4)
    out=projective_sandwich_test(s,t,mapping,tol=1e-6)
    assert out['ok'], out


def test_self_corpus_survives_strong_filters_without_direct_solver():
    s=load_scheme_json(CERT); assert s is not None
    t,_=_transform(s, 4)
    out=search(s,[t],direct_limit=0)
    assert out['stage_counts']['same_full_rank_pattern'] == 1
    assert out['stage_counts']['same_wl_incidence'] == 1


def test_kauers_exp_parser_content_based(tmp_path):
    p = tmp_path / "renamed_scheme.exp"
    p.write_text("\n".join(["(a11)*(b11)*(c11)"] * 23) + "\n")
    s = load_scheme_exp(p)
    assert s is not None
    assert s.rank == 23
    A, B, C = s.factors
    assert A[0] == sp.Matrix([[1,0,0],[0,0,0],[0,0,0]])
    assert B[0] == A[0]
    assert C[0] == A[0]


def test_kauers_exp_parser_bakes_scalar_into_first_leg():
    A, B, C = _split_exp_product("(3*(a11-a22))*(b11+b12)*(c33)/9")
    assert A == sp.Matrix([[sp.Rational(1,3),0,0],[0,sp.Rational(-1,3),0],[0,0,0]])
    assert B == sp.Matrix([[1,1,0],[0,0,0],[0,0,0]])
    assert C == sp.Matrix([[0,0,0],[0,0,0],[0,0,1]])


def test_corpus_audit_accounts_for_skips(tmp_path):
    from rank23_equivalence_search import scan_corpus_detailed
    # One supported 23-summand .exp file.
    good = tmp_path / "k-test-333-23-mod0.exp"
    good.write_text("\n".join(["(a11)*(b11)*(c11)"] * 23) + "\n")
    # Explicit wrong rank.
    (tmp_path / "wrong.exp").write_text("\n".join(["(a11)*(b11)*(c11)"] * 22) + "\n")
    # A plausible rank-23 file in a format we do not support must not disappear.
    (tmp_path / "mystery-333-23.scheme").write_text("some scheme\n")
    # Metadata JSON should be diagnosed by schema rather than ignored.
    (tmp_path / "metadata.json").write_text('{"hello": "world"}\n')
    # Reduced Perminov-style placeholder is an explicit coverage category.
    (tmp_path / "3x3x3_m23_reduced.json").write_text('{"reduced": true, "n": [3,3,3], "m": 23}\n')

    schemes, report, records = scan_corpus_detailed(tmp_path)
    assert len(schemes) == 1
    assert report["parsed_rank23"] == 1
    assert report["status_counts"]["wrong_rank"] == 1
    assert report["status_counts"]["plausible_unsupported_extension"] == 1
    assert report["status_counts"]["unsupported_json_schema"] == 1
    assert report["status_counts"]["unsupported_reduced_json"] == 1
    assert len(records) == 5


def test_write_corpus_audit_outputs_ledger(tmp_path):
    from rank23_equivalence_search import scan_corpus_detailed, write_corpus_audit
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "mystery-333-23.foo").write_text("x")
    _, report, records = scan_corpus_detailed(corpus)
    out = tmp_path / "audit"
    write_corpus_audit(out, [report], [records], sample_limit=2)
    assert (out / "CORPUS_AUDIT.md").exists()
    assert (out / "corpus_audit.json").exists()
    csv_text = (out / "corpus_files.csv").read_text()
    assert "plausible_unsupported_extension" in csv_text
