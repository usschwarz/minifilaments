'''
Written by Tom Kaufmann (Institute for Theoretical Physics and BioQuant, Heidelberg University, Heidelberg, Germany) 2019.

Correspondence should be addressed to Prof. Ulrich S. Schwarz at schwarz@thphys.uni-heidelberg.de.
'''

from utils import get_optimum_bending, calc_pairwise_potential
import argparse
import numpy as np
from datetime import datetime

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--charges1", required=True)
    ap.add_argument("--charges2", required=True)
    ap.add_argument('--antiparallel', dest='antiparallel', action='store_const', const=True, default=False)
    args = vars(ap.parse_args())

    antiparallel = args['antiparallel']
    anti_str = ''
    if antiparallel:
        anti_str = 'anti_'

    # Open charges
    with open(args['charges1'], 'rb') as f:
        charges1 = np.load(f)
    iso1 = args['charges1'].split('_')[0].split('/')[2]
    with open(args['charges2'], 'rb') as f:
        charges2 = np.load(f)
    iso2 = args['charges2'].split('_')[0].split('/')[2]

    potentials = np.zeros((1000, 200))

    for i in range(1000//5): # unbind
        # Add the normal (i.e. straight-line) potential
        normal_potential = calc_pairwise_potential(charges1,
                                charges2[:i*5],
                                antiparallel=antiparallel)

        normal_potential = normal_potential[(len(normal_potential)//2):(len(normal_potential)//2 + 1000)]
        # has to be this way because some mutants are smaller than 1000aas
        potentials[:len(normal_potential), i] = normal_potential

        for x in range(0, 1000, 1): # x-stagger
            potentials[x, i] += get_optimum_bending(charges1, charges2, x, i*5, antiparallel=antiparallel)[2]

    np.save("results/potentials/potentials_" + anti_str + "x_unfold_{}_{}.npy".format(iso1, iso2), 
        potentials)

    print('Done with potential: {}{}_{} at {}'.format(anti_str, iso1, iso2, str(datetime.now()).split(' ')[0]))