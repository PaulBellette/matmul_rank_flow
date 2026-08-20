#!/usr/bin/env python3
import json
from pathlib import Path
R=27
U=[[['0'] for _ in range(R)] for _ in range(9)]
V=[[['0'] for _ in range(R)] for _ in range(9)]
W=[[['0'] for _ in range(R)] for _ in range(9)]
c=[['1'] for _ in range(R)]
r=0
for i in range(3):
  for j in range(3):
    for k in range(3):
      U[3*i+j][r]=['1']; V[3*j+k][r]=['1']; W[3*i+k][r]=['1']; r+=1
d={'rank':R,'field':'Q','field_degree':1,'number_field':{'generator':'alpha','degree':1,'minimal_polynomial_coefficients_ascending':['0','1'],'embedding_approx':'0'},'U_power_basis':U,'V_power_basis':V,'W_power_basis':W,'c_power_basis':c}
Path('schoolbook_rank27.json').write_text(json.dumps(d,indent=2)+'\n')
print('schoolbook_rank27.json')
