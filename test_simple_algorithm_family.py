import random

from simple_algorithm_family import multiply, z_from_t


def schoolbook(A,B):
    return [
        [A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
        [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]],
    ]


def test_z_relation():
    for t in [0.0,0.01,0.2,0.5,0.9,1.0]:
        z=z_from_t(t)
        assert abs(t*t*(z*z-z+1)-z) < 1e-12


def test_simple_family_random_numeric():
    rng=random.Random(0)
    for t in [0.0,0.1,0.4,0.8,1.0]:
        for _ in range(20):
            A=[[rng.uniform(-2,2),rng.uniform(-2,2)],[rng.uniform(-2,2),rng.uniform(-2,2)]]
            B=[[rng.uniform(-2,2),rng.uniform(-2,2)],[rng.uniform(-2,2),rng.uniform(-2,2)]]
            got=multiply(A,B,t)
            want=schoolbook(A,B)
            assert max(abs(got[i][j]-want[i][j]) for i in range(2) for j in range(2)) < 2e-12
