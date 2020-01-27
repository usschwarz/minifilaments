'''
Written by Tom Kaufmann (Institute for Theoretical Physics and BioQuant, Heidelberg University, Heidelberg, Germany) 2019.

Correspondence should be addressed to Prof. Ulrich S. Schwarz at schwarz@thphys.uni-heidelberg.de.
'''

from itertools import product

ISOS = ['A', 'B', 'C', 'M18A']

final_files = ['results/T1/T1{}x_unfold_{}_{}.png'.format(anti_str, a, b) for anti_str in ['_anti_', '_'] for a, b in product(ISOS, repeat=2)] 

rule all:
    input:
        final_files

rule create_all_charges:
    input:
        expand(["data/charges/{iso}_charges.npy"], iso=ISOS)

rule calc_T1:
    input:
        script = "scripts/calc_T1.py",
        potentials = "results/potentials/potentials{anti}x_unfold_{iso1}_{iso2}.npy",
        charges1 = "data/charges/{iso1}_charges.npy"

    output:
        "results/T1/T1{anti}x_unfold_{iso1}_{iso2}.npy",
    run:
        if wildcards.anti == '_anti_':
            shell("python {input.script} --potentials {input.potentials} --charges1 {input.charges1} --antiparallel")
        else:
            shell("python {input.script} --potentials {input.potentials} --charges1 {input.charges1}")
            

rule calc_potentials_x_unfold:
    input:
        script = "scripts/calc_potentials_x_unfold.py",
        charges1 = "data/charges/{iso1}_charges.npy",
        charges2 = "data/charges/{iso2}_charges.npy"
    output:
        "results/potentials/potentials{anti}_x_unfold_{iso1}_{iso2}.npy"
    run:
        if wildcards.anti == '_anti_':
            shell("python {input.script} --charges1 {input.charges1} --charges2 {input.charges2} --antiparallel")
        else:
            shell("python {input.script} --charges1 {input.charges1} --charges2 {input.charges2}")



rule calc_charges:
    input:
        script = "scripts/calc_charges.py",
        seq = "data/seqs/{iso}.fasta",
        paircoil = "data/paircoil/{iso}_paircoil_clean.txt",
        parameters = "data/isoform_parameters.json"
    output:
        "data/charges/{iso}_charges.npy"
    shell:
        "python {input.script} --seq {input.seq} --paircoil {input.paircoil}"

rule clean_paircoil:
    input:
        script = "scripts/clean_paircoil.py",
        data = "data/paircoil/{iso}_paircoil.txt"
    output:
        "data/paircoil/{iso}_paircoil_clean.txt"
    shell:
        "python {input.script} --input {input.data} --output {output}"

rule paircoil:
    input:
        "data/seqs/{iso}.fasta"
    output:
        "data/paircoil/{iso}_paircoil.txt"
    shell:
        "./paircoil2 {input} {output}"

