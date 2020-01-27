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
        paircoil_raw = f.readlines()

    start = 3
    if paircoil_raw[start].split()[0] != '1':
        start = 4

    paircoil = []
    for line in paircoil_raw[start:]:
        paircoil.append(line.split())
    paircoil = np.array(paircoil)

    register = paircoil[:, 2]
    p_vals = paircoil[:, 3].astype(float)

    register[np.where(p_vals > 0.05)[0]] = '.'

    with open(args['output'], 'w') as f:
        f.write(''.join(register))
        