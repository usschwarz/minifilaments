'''
Written by Tom Kaufmann (Institute for Theoretical Physics and BioQuant, Heidelberg University, Heidelberg, Germany) 2019.

Correspondence should be addressed to Prof. Ulrich S. Schwarz at schwarz@thphys.uni-heidelberg.de.

All functions use the length unit of 1aa = 0.1456nm which corresponds to the 
distance between two neighboring amino acids in the regular coiled-coil.
The energy scale used is k_B T.

`k` always represents the inverse Debye-Hueckel length k = 1/1.3nm.
`y` represents the lateral distance between the two rods (y = 14aa = 2nm).
'''

import numpy as np
from numba import jit


def calc_straight_potential(charges_1,
                            charges_2,
                            y=14,
                            antiparallel=False,
                            k=0.11):
    '''
    Calculate the electrostatic potential of two straight rods.
    This is done using convolutions (immense speed-up over 2 for-loops)

    `charges_2` is the rod shifted by stagger s
    '''

    r = np.abs(np.arange(-200, 200))
    r = np.sqrt(r**2 + y**2)
    kernel = np.exp(-k * r) / r

    if not antiparallel:
        charges_1 = charges_1[::-1]

    pot = np.convolve(np.pad(charges_1, pad_width=100, mode='constant', constant_values=0),
                      kernel, 
                      mode='full')

    result = np.convolve(np.pad(charges_2, pad_width=100, mode='constant', constant_values=0),
                         np.pad(pot, pad_width=100,
                                mode='constant', constant_values=0),
                         mode='full')

    pad_width = np.abs(len(charges_1) - len(charges_2))
    result = np.pad(result, 
                    pad_width=pad_width,
                    mode='constant', 
                    constant_values=0)

    midpoint = 3*100 + len(kernel)//2 + 2 * pad_width + len(charges_2)
    # Value 4.75 to transform to units of [k_B T]
    result = 4.75 * result[(midpoint-len(charges_1)):(midpoint+len(charges_1))]

    return result[::-1]


def calc_bending_energy(L, R, lp=890, return_in_kBT=True):
    '''
    Calculate the bending energy according to the worm-like-chain model for
    polymers with locally straight conformation. The energy is 
    $$
    E_{\text{bend}} = \frac{l_p L}{2 R^2} k_B T
    $$
    '''

    if return_in_kBT:
        return lp * L / (2 * R**2)
    else:
        # Return in Joule
        return lp * L / (2 * R**2) * 4.11e-21


@jit(nopython=True)
def calc_arc_potential(charges1,
                       charges2,
                       stagger,
                       L_o,
                       radius,
                       L_a,
                       y=14,
                       k=0.11,
                       antiparallel=False):
    '''
    This function calculates the electrostatic energy between the bent part of 
    charges_2 and all of charges_1.
    It consists of two parts. First the actually bent part of the rod (of length
    L_a, called `potential_arc`) and the part after the bent part (of length
    total_length - L_o - L_a, called `potential_arc`).

    `L_o` and `L_a` correspond to the straight overlap between the two rods and
    the arc length, respectively (see the paper for more information).    

    `charges2` represents the bent rod and `charges_1` the straight one.
    In the antiparallel case, `charges1` is the reversed.
    '''

    window = int(2/k)
    quarter_circ = 2 * np.pi * radius / 4
    L_o = int(L_o)
    L_a = int(L_a)

    potential_arc = 0.
    potential_slope = 0.

    if antiparallel:
        charges1 = charges1[::-1]

    if stagger + L_o < len(charges1)+window:
        # i is the index for the bend part
        for i in range(0, min(len(charges2) - L_o, L_a)):
            curr_charges2 = charges2[i+L_o]
            if curr_charges2:
                curr_x = stagger + L_o + (radius * np.sin(np.pi/2 * i / quarter_circ))
                curr_y = y + radius * (1 - np.cos(np.pi/2 * i / quarter_circ))
                if (curr_x > len(charges1) + window or curr_y > y + window):
                    break
                # j is the index of the straight one, curr_j is the middle one
                curr_j = int(curr_x)
                for j in range(max(0, curr_j - window), min(len(charges1), curr_j + window)):
                    if charges1[j]:
                        r = np.sqrt((j-curr_j)**2 + (curr_y)**2)
                        potential_arc += curr_charges2 * \
                            charges1[j] * np.exp(-r * k)/r
    
    # i is the index for the bend part
    sin0 = np.sin(np.pi/2 * L_a / quarter_circ)
    cos0 = np.cos(np.pi/2 * L_a / quarter_circ)
    x0 = stagger + L_o + radius * sin0
    y0 = y + radius * (1 - cos0)

    if x0 < len(charges1)+window:

        for i in range(0, len(charges2) - L_o - L_a):
            curr_charges2 = charges2[i+L_a+L_o]
            if curr_charges2:
                curr_x = x0 + i * cos0
                curr_y = y0 + i * sin0
                if (curr_x > len(charges1) + window or curr_y > y + window):
                    break

                curr_j = int(curr_x)
                for j in range(max(0, curr_j - window), min(len(charges1), curr_j + window)):
                    if charges1[j]:
                        r = np.sqrt((j-curr_j)**2 + (curr_y)**2)
                        potential_slope += curr_charges2 * \
                            charges1[j] * np.exp(-r * k)/r

    # Adjust for kBT
    potential_arc *= 4.75
    potential_slope *= 4.75

    return potential_arc + potential_slope


def get_optimum_bending(charges1,
                        charges2,
                        stagger,
                        L_o,
                        antiparallel=False, 
                        k=0.11):
    '''
    Get optimum bending does NOT consider the straight part as it does not 
    matter for the bending (i.e. it's the same for all R and L_a!).
    It will return the optimum energy for a given stagger and overlap L_o which will 
    include the bending energy as well as the energy of the bending and sloping
    part of the electrostatic energy but not the straight part of the 
    electrostatic energy (which has to be added one level higher!)
    '''
    Rs = np.arange(500, 2000, 100)
    L_as = np.arange(100, 250, 10)
    energies = np.zeros((len(Rs), len(L_as)))

    for i, R in enumerate(Rs):
        for j, L_a in enumerate(L_as):
            energies[i, j] = calc_bending_energy(L=L_a, R=R) + \
                             calc_arc_potential(charges1, charges2, stagger=stagger, 
                                                L_o=L_o, radius=R, k = k,
                                                L_a=L_a, antiparallel=antiparallel)
    
    minimum_values = np.unravel_index(np.argmin(energies), energies.shape)
    return Rs[minimum_values[0]], L_as[minimum_values[1]], energies[minimum_values[0], minimum_values[1]]


def calc_T1(pot, starting_overlap, a=0, b=None, D=1):
    '''
    Calculate the mean contact time based on the mean first passage time in 
    the Fokker-Planck framework given the potential `pot` (in units of kB T)
    $$
    T(x) = \frac{1}{D} \int_x^b dz \, \exp \left(\frac{V(z)}{k_B T} \right) \left[\int_a^z dy \, \exp \left(- \frac{V(y)}{k_B T} \right) \right]
    $$
    '''

    if not b:
        b = len(pot)

    T1 = 0
    for z in range(starting_overlap, b):
        T1 += np.exp(pot)[z] * (z-a) * np.mean(np.exp(-pot[a:z]/D))

    T1 *= (b-starting_overlap) * 1/D

    return T1
