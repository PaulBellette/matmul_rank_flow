# Five-seed rank-23 endpoint analysis

- endpoints analysed: **5/5**
- stable canonical rank-pattern classes: **4**
- all endpoints stable across tolerances: **False**

| seed | class | input residual | polished residual | max |a| | stable pattern? | stable WL? | pattern hash |
|---:|:---:|---:|---:|---:|:---:|:---:|:---|
| 101 | C1 | 2.380e-13 | 3.837e-15 | 5.138 | YES | YES | `87c8fa24dd51f45d` |
| 211 | C2 | 3.608e-15 | 3.608e-15 | 4.608 | YES | YES | `42543f8d7faf9c9b` |
| 307 | UNSTABLE | 3.107e-10 | 6.449e-15 | 3.856 | NO | NO | `83ab344dd447d890` |
| 401 | C3 | 8.306e-10 | 2.682e-15 | 3.572 | YES | YES | `c09c8dd9c5b941e3` |
| 503 | C4 | 3.229e-15 | 3.229e-15 | 3.593 | YES | YES | `b4c5eec02a784e23` |

## Structural classes

- **C1**: seeds [101]; factor ranks `{'0': 0, '1': 43, '2': 25, '3': 1}`; channel triples `{'(1, 1, 1)': 7, '(1, 1, 2)': 4, '(1, 1, 3)': 1, '(1, 2, 1)': 3, '(1, 2, 2)': 2, '(2, 1, 1)': 2, '(2, 2, 2)': 4}`; JKU same canonical pattern: **0**
- **C2**: seeds [211]; factor ranks `{'0': 0, '1': 44, '2': 24, '3': 1}`; channel triples `{'(1, 1, 1)': 9, '(1, 1, 2)': 2, '(1, 1, 3)': 1, '(1, 2, 1)': 1, '(1, 2, 2)': 2, '(2, 1, 1)': 3, '(2, 2, 1)': 1, '(2, 2, 2)': 4}`; JKU same canonical pattern: **0**
- **C3**: seeds [401]; factor ranks `{'0': 0, '1': 46, '2': 22, '3': 1}`; channel triples `{'(1, 1, 1)': 8, '(1, 1, 2)': 3, '(1, 2, 1)': 3, '(2, 1, 1)': 4, '(2, 2, 2)': 4, '(3, 1, 1)': 1}`; JKU same canonical pattern: **7**
- **C4**: seeds [503]; factor ranks `{'0': 0, '1': 42, '2': 24, '3': 3}`; channel triples `{'(1, 1, 1)': 7, '(1, 1, 2)': 3, '(1, 1, 3)': 2, '(2, 1, 1)': 4, '(2, 2, 1)': 2, '(2, 2, 2)': 4, '(2, 3, 1)': 1}`; JKU same canonical pattern: **0**
- **UNSTABLE**: seeds [307]

The frozen controller reached multiple distinct canonical factor-rank patterns.
Once exactified, differing stable patterns would prove the endpoints lie in inequivalent isotropy orbits.

## JKU archive comparison

- parsed `.exp` schemes: **17376**
- parse errors: **0**

Archive counts use exact integer factor ranks. Endpoint classes are still numerical until exactified,
so a zero match is strong classification evidence but should be phrased conservatively until the endpoint is exactified.
