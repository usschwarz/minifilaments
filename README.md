# Staggering and splaying in nonmuscle myosin II minifilaments
Code for the manuscript **Staggering and splaying in nonmuscle myosin II minifilaments** by Tom Kaufmann and Ulrich S. Schwarz (*Institute for Theoretical Physics and BioQuant, Heidelberg University, Heidelberg, Germany*).

Correspondence should be addressed to Ulrich Schwarz at schwarz@thphys.uni-heidelberg.de. ([Website](https://www.thphys.uni-heidelberg.de/~biophys/))


## Folder structure
The project requires the folder structure as shown below
```
├── data
│   ├── charges
│   ├── paircoil
│   ├── seqs
├── results
│   ├── potentials
│   └── T1
├── scripts
```

This structure can be created using the command `mkdir -p data/{paircoil,charges} results/{potentials,T1}` on *nix machines.


## Workflow
The workflow is run using the workflow manager [Snakemake](https://snakemake.readthedocs.io/en/stable/).

### Scripts
All scripts are in the folder `scripts` and use functions from the file `utils.py`.

### Data
The amino acid sequences of the NM2 heavy chains found in the folder `data/seqs` were obtained from the [NCBI Protein Database](https://www.ncbi.nlm.nih.gov/protein) with the accession numbers: *NM2A - P35579*, *NM2B - P35580*, *NM2C - NP_079005* and *M18A - Q92614*.
All other data are created in the workflow.

### Paircoil
The workflow uses the programm [Paircoil2](https://academic.oup.com/bioinformatics/article/22/3/356/220410).
Proper execution of the workflow requires the program the paircoil binary `paircoil2` as well as the reference data `nr90-050325.tb` and `newcc.tb` from the [Official paircoil website](http://cb.csail.mit.edu/cb/paircoil2/) placed in the main folder.
