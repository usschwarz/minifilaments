'''
Written by Tom Kaufmann (Institute for Theoretical Physics and BioQuant, Heidelberg University, Heidelberg, Germany) 2019.

Correspondence should be addressed to Prof. Ulrich S. Schwarz at schwarz@thphys.uni-heidelberg.de.
'''

import numpy as np
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = vars(ap.parse_args())

    with open(args['input'], 'r') as f:
        pc = f.readlines()

    start = 3
    if pc[start].split()[0] != '1':
        start = 4

    pc_ = []
    for p in pc[start:]:
        pc_.append(p.split())
    pc = np.array(pc_)

    register = pc[:, 2]
    p_vals = pc[:, 3].astype(float)

    register[np.where(p_vals > 0.05)[0]] = '.'

    with open(args['output'], 'w') as f:
        f.write(''.join(register))