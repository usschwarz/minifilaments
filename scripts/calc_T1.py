'''
Written by Tom Kaufmann (Institute for Theoretical Physics and BioQuant, Heidelberg University, Heidelberg, Germany) 2019.

Correspondence should be addressed to Prof. Ulrich S. Schwarz at schwarz@thphys.uni-heidelberg.de.
'''

import argparse
import numpy as np
from matplotlib import pyplot as plt
from utils import calc_T1

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--potentials", required=True)
    ap.add_argument("--charges1", required=True)
    ap.add_argument('--antiparallel', dest='antiparallel', action='store_const', const=True, default=False)
    args = vars(ap.parse_args())

    antiparallel = args['antiparallel']
    anti_str = ''
    if antiparallel:
        anti_str = '_anti'

    potentials = np.load(args['potentials'])
    with open(args['charges1'], 'rb') as f:
        charges1 = np.load(f)
    iso1 = args['potentials'].split('.')[0].split('_')[-2]
    iso2 = args['potentials'].split('.')[0].split('_')[-1]

    T1s = []
    for x in range(0, 1000):
        potential = potentials[x]
        # correction of (len(charges1)-1000)//5 because we only until 1000aas
        # but the charges are actually longer (practically very little difference)
        T1s.append([calc_T1(unzip, potential[::-1], a = max(0, x//5  - (len(charges1)-1000)//5)) if potential[200-unzip] < 0 else 0. for unzip in [190, 195]])
    T1s = np.array(T1s)

    np.save('results/T1/T1' + anti_str + '_x_unfold_{}_{}.npy'.format(iso1, iso2), T1s)

    # Plotting
    fig, ax = plt.subplots()

    ax.plot(T1s[:, 0]/1000, color = 'C1', label = 'initial contact = 50 aas')
    ax.plot(T1s[:, 1]/1000, color = 'C2', label = 'initial contact = 25 aas')

    plt.legend()

    for peak in [144/1.456, 432/1.456, 720/1.456]:
        plt.axvline(peak, color = 'grey', alpha = 0.4)

    plt.savefig('results/T1/T1' + anti_str + '_x_unfold_{}_{}.png'.format(iso1, iso2))