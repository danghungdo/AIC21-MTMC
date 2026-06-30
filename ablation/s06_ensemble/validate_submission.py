import numpy as np
from collections import defaultdict
import sys

T = sys.argv[1] if len(sys.argv) > 1 else 'reid/reid-matching/tools/track3.txt'
rows = [l.split() for l in open(T)]
a = np.array([[float(x) for x in r] for r in rows])
print('rows:', len(rows), 'col-counts:', set(len(r) for r in rows))
print('cameras:', sorted(set(a[:, 0].astype(int))))
print('obj ids: min', int(a[:, 1].min()), 'max', int(a[:, 1].max()),
      'positive-int:', bool((a[:, 1] >= 1).all() and (a[:, 1] == a[:, 1].astype(int)).all()))
print('frame ids: min', int(a[:, 2].min()), 'max', int(a[:, 2].max()), '(must start >=1)')
print('bbox w[%d,%d] h[%d,%d]  any w/h<=0:' % (a[:, 5].min(), a[:, 5].max(), a[:, 6].min(), a[:, 6].max()),
      bool((a[:, 5] <= 0).any() or (a[:, 6] <= 0).any()))
print('world coords col8/col9 values:', sorted(set(a[:, 7].astype(int)))[:3], sorted(set(a[:, 8].astype(int)))[:3])
idcams = defaultdict(set)
for r in a:
    idcams[int(r[1])].add(int(r[0]))
print('global IDs total:', len(idcams), ' scored (>=2 cameras):', sum(len(v) >= 2 for v in idcams.values()))
