'''
Written by Tom Kaufmann (Institute for Theoretical Physics and BioQuant, Heidelberg University, Heidelberg, Germany) 2019.

Correspondence should be addressed to Prof. Ulrich S. Schwarz at schwarz@thphys.uni-heidelberg.de.
'''

import numpy as np
import argparse
import json

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--paircoil", required=True)
    args = vars(ap.parse_args())

    iso = args['seq'].split('.')[0].split('/')[2]

    with open("data/isoform_parameters.json") as f:
        isoform_parameters = json.load(f)[iso]

    with open(args['seq'], 'r') as f:
        seq = f.readlines()
    seq = np.array(list(seq[1].rstrip()))

    with open(args['paircoil'], 'r') as f:
        paircoil = f.readlines()
    paircoil = np.array(list(paircoil[0].rstrip()))

    assert len(seq) == len(paircoil), \
        "For iso {}, seq and paircoil do not have the same lengths\nlen(seq)={}, len(paircoil)={}".format(
            iso, len(seq), len(paircoil))

    charges = np.zeros(len(seq))

    charges[seq == 'R'] = 1  # Arganine
    charges[seq == 'K'] = 1  # Lysine
    charges[seq == 'D'] = -1  # Aspartic acid
    charges[seq == 'E'] = -1  # Glutamic acid

    charges[paircoil == 'a'] = 0
    charges[paircoil == 'd'] = 0

    charges = charges[isoform_parameters['start']:isoform_parameters['end']]
    charges *= 2
    charges = charges[::-1]

    np.save('data/charges/' + iso + '_charges.npy', charges)
