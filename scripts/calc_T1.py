'''
Written by Tom Kaufmann (Institute for Theoretical Physics and BioQuant, Heidelberg University, Heidelberg, Germany) 2019.

Correspondence should be addressed to Prof. Ulrich S. Schwarz at schwarz@thphys.uni-heidelberg.de.
'''

import argparse
import numpy as np
from utils import calc_T1

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--potentials", required=True)
    ap.add_argument("--charges1", required=True)
    ap.add_argument('--antiparallel', dest='antiparallel', 
                    action='store_const', const=True, default=False)
    args = vars(ap.parse_args())

    anti_str = ''
    if args['antiparallel']:
        anti_str = '_anti'

    with open(args['charges1'], 'rb') as f:
        charges1 = np.load(f)
    potentials = np.load(args['potentials'])
    iso1 = args['potentials'].split('.')[0].split('_')[-2]
    iso2 = args['potentials'].split('.')[0].split('_')[-1]

    T1s = []
    for s in range(0, 1000):
        potential = potentials[s]
        # correction of `max(0, s//5  - (len(charges1)-1000)//5))` because we 
        # only calculate until 1000aas but the charges are actually longer 
        current_T1 = calc_T1(starting_overlap = 195, 
                            pot = potential[::-1], 
                            a = max(0, s//5  - (len(charges1)-1000)//5), 
                            b = len(potential)) 
        T1s.append(current_T1)
    
    T1s = np.array(T1s)

    np.save('results/T1/T1' + anti_str + '_{}_{}.npy'.format(iso1, iso2), T1s)
