'''
Written by Tom Kaufmann (Institute for Theoretical Physics and BioQuant, Heidelberg University, Heidelberg, Germany) 2019.

Correspondence should be addressed to Prof. Ulrich S. Schwarz at schwarz@thphys.uni-heidelberg.de.
'''

import numpy as np
from matplotlib import pyplot as plt
from numba import jit


@jit(nopython=True)
def calc_pairwise_potential_OLD(charges_1, charges_2, x, y,
                                antiparallel=False, k=0.11):
    '''
    DEPRICATED (is a lot slower and unaccurate than new version)
    Loop along charges_2
    '''

    potential = 0
    # Only take 2*1/k charges into account
    window = int(2/k)
    if antiparallel:
        charges_1 = charges_1[::-1]

    for j in range(len(charges_2)):
        for i in range(max(0, j+x-window), min(len(charges_1), j+x+window+1)):
            if charges_1[i] and charges_2[j]:
                r = np.sqrt((i-(x+j))**2 + y**2)
                potential += charges_1[i] * charges_2[j] * np.exp(-r * k)/r

    # Tune to the correct scale
    potential *= 4.75

    return potential


def calc_pairwise_potential(charges_1,
                            charges_2,
                            y=14,
                            kernel=None,
                            antiparallel=False,
                            k=0.11):
    '''
    charges_2 is always the shorter one!
    '''

    if not kernel:
        r = np.abs(np.arange(-200, 200))
        r = np.sqrt(r**2 + (y)**2)
        kernel = np.exp(-k * r) / r

    if not antiparallel:  # It's correct this way!
        charges_1 = charges_1[::-1]

    pot = np.convolve(np.pad(charges_1, pad_width=100, mode='constant', constant_values=0),
                      kernel, mode='full')

    result = np.convolve(np.pad(charges_2, pad_width=100, mode='constant', constant_values=0),
                         np.pad(pot, pad_width=100,
                                mode='constant', constant_values=0),
                         mode='full')

    pad_width = np.abs(len(charges_1) - len(charges_2))

    result = np.pad(result, pad_width=pad_width,
                    mode='constant', constant_values=0)
    midpoint = 3*100 + len(kernel)//2 + 2 * pad_width + len(charges_2)
    result = 4.75 * result[(midpoint-len(charges_1)):(midpoint+len(charges_1))]

    result = result[::-1]

    return result


def plot_two_myosin(ch,
                    x,
                    unzip,
                    radius,
                    extend_bend=-1,
                    show_charges_color=False,
                    x_zoom=None,
                    antiparallel=False,
                    fig=None,
                    ax=None):
    if not ax or not fig:
        fig, ax = plt.subplots(figsize=(15, 5))

    if show_charges_color:
        color_dict = {0: 'grey', 1: 'blue', -1: 'red'}
        colors = [[color_dict[c] for c in ch], [color_dict[c] for c in ch]]

    else:
        colors = [len(ch)*['green'], len(ch)*['purple']]

    x = int(x)
    unzip = int(unzip)
    radius = int(radius)
    extend_bend = int(extend_bend)

    circ_4 = int(np.pi * radius / 2)
    rest_chain = np.min([circ_4, len(ch)-unzip])

    if antiparallel:
        ax.set_ylim(-25, 1 + radius *
                    (1 - np.cos(np.pi/2 * (len(ch) - unzip) / circ_4)) + 25)
        ax.set_xlim(-1 * len(ch) + x, len(ch)+25)
    else:
        ax.set_xlim(-25, x+len(ch)+25)
        if extend_bend == -1 or extend_bend > (len(ch) - unzip):
            ax.set_ylim(-25, 1 + radius * (1 - np.cos(np.pi /
                                                      2 * (len(ch) - unzip) / circ_4)) + 25)
        else:
            ax.set_ylim(-25, 14 + radius * (1 - np.cos(np.pi/2 * extend_bend / circ_4)) + (
                len(ch)-unzip-extend_bend) * np.sin(np.pi/2 * extend_bend / circ_4) + 25)

    width = ax.figure.bbox_inches.width * ax.get_position().width

    xlims = np.diff(ax.get_xlim())
    inch_per_scale = width / xlims
    markersize = 7 * (inch_per_scale * 72 * 2)

    # Actual plotting
    ax.scatter(np.arange(0, len(ch)), len(ch) *
               [0], color=colors[0], s=markersize**2)

    if antiparallel:
        ax.scatter(np.arange(x-unzip, x), unzip *
                   [14], color=colors[1][:unzip], s=markersize**2)

        ax.scatter(x-unzip - (radius * np.sin(np.pi/2 * np.arange(0, len(ch) - unzip) / circ_4)),
                   14 + radius *
                   (1 - np.cos(np.pi/2 * np.arange(0, len(ch) - unzip) / circ_4)),
                   color=colors[1][unzip:], s=markersize**2)
        ax.scatter(x-unzip - radius * np.sin(np.pi/2 * (len(ch) - unzip) / circ_4),
                   14 + radius *
                   (1 - np.cos(np.pi/2 * (len(ch) - unzip) / circ_4)),
                   s=2000, color='black')

        ax.axvline(x-unzip, color='black')
        ax.axvline(x, color='black')

    else:
        ax.scatter(np.arange(x, x+unzip), unzip *
                   [14], color=colors[1][:unzip], s=markersize**2)

        if extend_bend == -1 or extend_bend > (len(ch) - unzip):
            ax.scatter(x+unzip + (radius * np.sin(np.pi/2 * np.arange(0, len(ch) - unzip) / circ_4)),
                       14 + radius *
                       (1 - np.cos(np.pi/2 * np.arange(0, len(ch) - unzip) / circ_4)),
                       color=colors[1][unzip:], s=markersize**2)

            ax.scatter(x+unzip + radius * np.sin(np.pi/2 * (len(ch) - unzip) / circ_4),
                       14 + radius *
                       (1 - np.cos(np.pi/2 * (len(ch) - unzip) / circ_4)),
                       s=2000, color='black')

        else:
            ax.scatter(x+unzip + (radius * np.sin(np.pi/2 * np.arange(0, extend_bend) / circ_4)),
                       14 + radius *
                       (1 - np.cos(np.pi/2 * np.arange(0, extend_bend) / circ_4)),
                       color=colors[1][unzip:unzip+extend_bend], s=markersize**2)
            ax.scatter(x+unzip + radius * np.sin(np.pi/2 * extend_bend / circ_4) + np.arange(0, len(ch)-unzip-extend_bend) * np.cos(np.pi/2 * extend_bend / circ_4),
                       14 + radius * (1 - np.cos(np.pi/2 * extend_bend / circ_4)) + np.arange(
                           0, len(ch)-unzip-extend_bend) * np.sin(np.pi/2 * extend_bend / circ_4),
                       color=colors[1][unzip+extend_bend:], s=markersize**2)
            ax.scatter(x+unzip + radius * np.sin(np.pi/2 * extend_bend / circ_4) + (len(ch)-unzip-extend_bend) * np.cos(np.pi/2 * extend_bend / circ_4),
                       14 + radius * (1 - np.cos(np.pi/2 * extend_bend / circ_4)) + (
                           len(ch)-unzip-extend_bend) * np.sin(np.pi/2 * extend_bend / circ_4),
                       s=2000, color='black')
            ax.scatter(x+unzip + radius * np.sin(np.pi/2 * extend_bend / circ_4),
                       14 + radius *
                       (1 - np.cos(np.pi/2 * extend_bend / circ_4)),
                       color='black', s=25)

        ax.scatter(x+unzip, 14, color='black', s=25)

    ax.scatter(len(ch), 0, s=2000, color='black')

    if x_zoom:
        ax.set_xlim(x_zoom[0], x_zoom[1])
    ax.set_aspect('equal')


def calc_bending_energy(L, R, lp=890, return_in_kbT=True):
    if return_in_kbT:
        return lp * L / (2 * R**2)
    else:
        # Return in J
        return lp * L / (2 * R**2) * 4.11e-21


@jit(nopython=True)
def calc_arc_potential(charges1,
                       charges2,
                       x,
                       unzip,
                       radius,
                       extend,
                       y=14,
                       k=0.11,
                       antiparallel=False,
                       skip_outside=True):
    '''
    Two contributions: arc, slope

    charges2 is always the bend one!
    charges1 is the reversed one for antiparallel (x = 0 is total overlap, x = len(charges1) is no overlap)
    '''

    window = int(2/k)
    circ_4 = np.pi * radius / 2
    unzip = int(unzip)
    extend = int(extend)

    potential_arc = 0.
    potential_slope = 0.

    if antiparallel:
        charges1 = charges1[::-1]

    if x + unzip < len(charges1)+window:
        # i is the index for the bend part
        for i in range(0, min(len(charges2) - unzip, extend)):
            curr_charges2 = charges2[i+unzip]
            if curr_charges2:
                curr_x = x + unzip + (radius * np.sin(np.pi/2 * i / circ_4))
                curr_y = y + radius * (1 - np.cos(np.pi/2 * i / circ_4))
                if (curr_x > len(charges1) + window or curr_y > y + window) and skip_outside:
                    break
                # j is the index of the straight one, curr_j is the middle one
                curr_j = int(curr_x)
                for j in range(max(0, curr_j - window), min(len(charges1), curr_j + window)):
                    if charges1[j]:
                        r = np.sqrt((j-curr_j)**2 + (curr_y)**2)
                        potential_arc += curr_charges2 * \
                            charges1[j] * np.exp(-r * k)/r

    
    
    # i is the index for the bend part
    sin0 = np.sin(np.pi/2 * extend / circ_4)
    cos0 = np.cos(np.pi/2 * extend / circ_4)
    x0 = x + unzip + radius * sin0
    y0 = y + radius * (1 - cos0)

    if x0 < len(charges1)+window:

        for i in range(0, len(charges2) - unzip - extend):
            curr_charges2 = charges2[i+extend+unzip]
            if curr_charges2:
                curr_x = x0 + i * cos0
                curr_y = y0 + i * sin0
                if (curr_x > len(charges1) + window or curr_y > y + window) and skip_outside:
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

#     return potential_arc, potential_slope
    return potential_arc + potential_slope


def get_optimum_bending(charges1,
                        charges2,
                        x,
                        unzip,
                        antiparallel=False, 
                        k=0.11):
    '''
    Get optimum bending does NOT consider the straight part as it does not 
    matter for the bending (i.e. it's the same for all R and extend!).
    It will return the optimum energy for a given x and unbing which will 
    include the bending energy as well as the energy of the bending and sloping
    part of the electrostatic energy but not the straight part of the 
    electrostatic energy (which has to be added one level higher!)
    '''
    rs = np.arange(500, 2000, 100)
    extends = np.arange(100, 250, 10)
    energies = np.zeros((len(rs), len(extends)))

    for i, r in enumerate(rs):
        for j, extend in enumerate(extends):
            energies[i, j] = calc_bending_energy(L=extend, R=r) + \
                             calc_arc_potential(charges1, charges2, x=x, 
                                                unzip=unzip, radius=r, k = k,
                                                extend=extend, antiparallel=antiparallel)
    
    minimum_values = np.unravel_index(np.argmin(energies), energies.shape)
    return rs[minimum_values[0]], extends[minimum_values[1]], energies[minimum_values[0], minimum_values[1]]


def get_attached_aas(len_ch, x, unzip, extend, radius, threshold=1.5):
    '''
    If pot0 is for r = y0 = 14aa, then at 1.5*y0 we have pot = 0.3 pot0 and at 2*y0 pot = 0.1 pot0

    check with 
    r0 = 14
    pot0 = np.exp(-r0 * k)/r0
    r = 2*r0
    np.exp(-r * k)/r / pot_0
    '''
    # threshold is in nm
    threshold *= 14
    circ_4 = int(np.pi * radius / 2)

    ys1 = 14 + radius * (1 - np.cos(np.pi/2 * np.arange(0, extend) / circ_4))
    ys2 = 14 + radius * (1 - np.cos(np.pi/2 * extend / circ_4)) + \
        np.arange(0, len_ch-unzip-extend) * np.sin(np.pi/2 * extend / circ_4)

    ys = np.concatenate([ys1, ys2])

    # The second part is the first position where the ys are larger than the thresold
    return unzip + np.where(ys > threshold)[0][0]


def calc_T1(x0, pot, a=0, b=None, D=1):
    '''
    For large x the function will give nan values.
    This is because the potentials are only of length 200, whilst they should be 1089//5 = 217 long. So it is cut off too early but that shouldn't matter too much as these big values are unimportant anyways
    '''

    if b == None:
        b = len(pot)

    T1 = 0
    for z in range(x0, b):
        T1 += np.exp(pot)[z] * (z-a) * np.mean(np.exp(-pot[a:z]/D))

    T1 *= (b-x0) * 1/D

    return T1
