This code requires following packages:

```
python3
numpy
pandas
matplotlib
MDAnalysis
parmed
```

## Get started
***topology_analysis_v9.3 USAGE***

```bash
python [path-to-python_script] [path-to-config-file] [outfile-basename] [in-path-for-trajectory-or-pdb] [start-frame] [end-frame] [frame-stride]
```

| Syntax      | Description |
| :---        |        ---: |
| path-to-config-file      | path to a .config file following the convention below       | 
| outfile-basename   | string used to form the basename of the output dataframe file| 
| in-path-for-trajectory-or-pdb | path to a [MDAnalysis supported trajectory file](https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html) or PDB file|
| start-frame | frame to start the analysis at |
| end-frame | frame to end the analysis at (non inclusive) |
| frame-stride | number of frames to skip between analysis |


***Example .config file and details***
```bash
[DEFAULT]

#input and output file paths and some tag and model resolution info
print_updatelog = yes
print_framesummary = yes
out_path = ./ 
protid = aa 
psf = ../traj/newbox.gro
# orig_aa_pdb should be a native structure 
orig_aa_pdb = setup/2ww4_chain_a_rebuilt_mini.pdb 
model_res = aa
bt_contact_pot_file = setup/bt_contact_potential.dat
cg_ref_crd = nan

#window size used for determining residues near entaglement
window_size = 5

#domains to analyze
dom_def = nan
sec_elems_file_path = setup/2ww4_sec_structure.txt
termini_threshold = 5,5

#analyze only the reference PDB
ref_only = False

#resid mask for fraction of native contacts, rmsd, and structural overlap analysis
#domain_name:'commda_sep_list_resids'-'df_label'
resid_masks = nan


#rmsd mask. either ([nan], [2struct], or [domain_name:'comma_sep_list_resids'])
rmsd_masks = nan

#method to use when finding entangled residues for native contacts
#with a non-zero total linking number [1,2]
#see Generation of entanglement metric distributions and use as an order parameter for protein folding.pdf
#for more details (default is 2)
ent_res_method = 2

#which analysis to do
[ANALYSIS]

fraction_native_contact_analysis = yes
fraction_native_contacts_wchngent_analysis = yes
entanglement_analysis = yes
rmsd_analysis = yes
struct_overlap_analysis = no

```

| Syntax      | Description |
| :---        |        ---: |
| DEFAULT | |
| print_updatelog | print the update log for the script |
| print_framesummary | print summary information for each frame to the stdout |
| out_path | path to workind directory where script will build output files |
| psf | path to protein structure file. Any MDAnalsyis supported topology file will work here not just CHARMM PSF |
| orig_aa_pdb | path to original all atom PDB used to create CG model or for start of all atom sims |
| bt_contact_pot_file | path to file containing condensed bt_contact_potential data |
| cg_ref_crd | path to reference cg coordiante file ending in .crd if analyzing a cg trajectory |
| model_res | aa or cg |
| window_size | number of residues used in the sliding window to determine entangled residues|
| dom_def | dictionary-like specified domain definitions. If nan parmed will split the structrue by chains|
| sec_elems_file_path | path to secondary structure index file generated from stride and formated as shown in example file |
| termini_threshold | number of residues to exclude from the ends of the termini when finding non-covalent lassos|
| ref_only | boolean flag for only analyzing the reference state and not the trajecotry. Does not actualy save a dataframe but streams usefull info to the screen | 
| resid_masks | if specified will over ride sec_elems_file for residues to find native contacts for. |
| rmsd_masks | provide either ["nan"], [a series of comma separated domain specific masks], or ["2structs" to only consider residues in 2nd structure elements]. |
| ent_res_method | (1) old method of finding ent residues, (2) new method that is more accurate taking into account whole loop topology. (3) same as method 2 but allows for the return of all g(i,j,m) values greater than a percentile specified by the user (see Generation_of_entanglement_metric_distributions_and_use_as_an_order_parameter_for_protein_folding_v2.1.pdf for more details) |
| ANALYSIS | |
| fraction_native_contact_analysis | boolean flag for outputting fraction_native_contact_analysis for each domain and total structure| 
| fraction_native_contact_wchngent_analysis | boolean flag for outputting fraction_native_contact_wchngent_analysis for each domain| 
| entanglement_analysis | boolean flag for outputting entanglement_analysis for each domain and between domains |
| rmsd_analysis | boolean flag for outputting rmsd_analysis for each domain and total structure |
| struct_overlap_analysis | boolean flag for calc and output of structural overlap function. Requires mask to be specified | 

***Example secondary structure definition***
```
Strand         2     20 A   A
Strand        26     46 A   A
Strand        51     53 A   A
310Helix      62     64 A   A
AlphaHelix    66     81 A   A
Strand        89     95 A   A
AlphaHelix   106    122 A   A
AlphaHelix   128    137 A   A
AlphaHelix   141    147 A   A
Strand       151    154 A   A
Strand       159    162 A   A
Strand       169    173 A   A
AlphaHelix   181    186 A   A
AlphaHelix   199    204 A   A
AlphaHelix   211    218 A   A
AlphaHelix   220    231 A   A
Strand       235    237 A   A
Strand       244    248 A   A
AlphaHelix   251    260 A   A
310Helix     263    265 A   A
Strand       267    273 A   A
AlphaHelix   277    282 A   A
```

column 1: arbitrary label

column 2: start residue of secondary structural element predicted by stride

column 3: end residue of secondary structural element predicted by stride

column 4: domain name in which secondary structural element is located (arbitrarly user defined)

column 5: chain name that matches topology file

***Notes on the format of the orig_aa_pdb***

The all-atom PDB used should have the following format:
```
ATOM      1  N   THR A   1      40.897  34.729  61.736  1.00 66.71      A    N
ATOM      2  HT1 THR A   1      40.323  33.966  62.149  1.00  0.00      A    H
ATOM      3  HT2 THR A   1      40.973  34.588  60.708  1.00  0.00      A    H
ATOM      4  HT3 THR A   1      40.444  35.645  61.927  1.00  0.00      A    H
ATOM      5  CA  THR A   1      42.264  34.711  62.345  1.00 66.65      A    C
ATOM      6  HA  THR A   1      42.121  34.943  63.392  1.00  0.00      A    H
ATOM      7  CB  THR A   1      42.904  33.290  62.254  1.00 66.85      A    C
ATOM      8  HB  THR A   1      42.081  32.565  62.469  1.00  0.00      A    H
ATOM      9  OG1 THR A   1      43.904  33.135  63.273  1.00 67.34      A    O
ATOM     10  HG1 THR A   1      44.260  32.248  63.150  1.00  0.00      A    H
ATOM     11  CG2 THR A   1      43.492  33.015  60.861  1.00 66.57      A    C
ATOM     12 HG21 THR A   1      43.840  31.964  60.779  1.00  0.00      A    H
ATOM     13 HG22 THR A   1      42.724  33.182  60.076  1.00  0.00      A    H
ATOM     14 HG23 THR A   1      44.355  33.684  60.652  1.00  0.00      A    H
ATOM     15  C   THR A   1      43.161  35.828  61.765  1.00 66.28      A    C
ATOM     16  O   THR A   1      42.725  36.575  60.887  1.00 66.38      A    O
ATOM     17  N   THR A   2      44.392  35.955  62.263  1.00 65.85      A    N
ATOM     18  HN  THR A   2      44.816  35.307  62.894  1.00  0.00      A    H
ATOM     19  CA  THR A   2      45.246  37.099  61.907  1.00 65.42      A    C
ATOM     20  HA  THR A   2      44.772  37.521  61.031  1.00  0.00      A    H
ATOM     21  CB  THR A   2      45.319  38.136  63.062  1.00 65.42      A    C
ATOM     22  HB  THR A   2      44.262  38.355  63.350  1.00  0.00      A    H
ATOM     23  OG1 THR A   2      45.908  39.356  62.587  1.00 64.94      A    O
ATOM     24  HG1 THR A   2      45.932  39.943  63.350  1.00  0.00      A    H
ATOM     25  CG2 THR A   2      46.142  37.600  64.234  1.00 65.19      A    C
ATOM     26 HG21 THR A   2      46.097  38.295  65.099  1.00  0.00      A    H
ATOM     27 HG22 THR A   2      45.748  36.616  64.566  1.00  0.00      A    H
ATOM     28 HG23 THR A   2      47.208  37.470  63.948  1.00  0.00      A    H
ATOM     29  C   THR A   2      46.677  36.740  61.497  1.00 65.21      A    C
ATOM     30  O   THR A   2      47.128  35.611  61.693  1.00 65.36      A    O
...

ATOM   4456  N   GLU A 284      83.464  36.018  75.690  1.00 58.81      A    N
ATOM   4457  HN  GLU A 284      82.510  36.225  75.890  1.00  0.00      A    H
ATOM   4458  CA  GLU A 284      84.076  35.068  76.624  1.00 60.80      A    C
ATOM   4459  HA  GLU A 284      85.089  34.864  76.301  1.00  0.00      A    H
ATOM   4460  CB  GLU A 284      84.136  35.623  78.056  1.00 60.96      A    C
ATOM   4461  HB1 GLU A 284      83.154  35.427  78.546  1.00  0.00      A    H
ATOM   4462  HB2 GLU A 284      84.890  35.058  78.648  1.00  0.00      A    H
ATOM   4463  CG  GLU A 284      84.434  37.119  78.173  1.00 62.01      A    C
ATOM   4464  HG1 GLU A 284      84.983  37.314  79.113  1.00  0.00      A    H
ATOM   4465  HG2 GLU A 284      85.042  37.471  77.316  1.00  0.00      A    H
ATOM   4466  CD  GLU A 284      83.168  37.968  78.250  1.00 62.92      A    C
ATOM   4467  OE1 GLU A 284      82.347  37.737  79.167  1.00 63.17      A    O
ATOM   4468  OE2 GLU A 284      82.998  38.872  77.402  1.00 62.86      A    O
ATOM   4469  C   GLU A 284      83.310  33.737  76.588  1.00 62.00      A    C
ATOM   4470  O   GLU A 284      83.408  32.980  75.619  1.00 62.19      A    O
ATOM   4471  N   GLY A 285      82.524  33.468  77.626  1.00 63.41      A    N
ATOM   4472  HN  GLY A 285      82.352  34.128  78.357  1.00  0.00      A    H
ATOM   4473  CA  GLY A 285      81.853  32.179  77.769  1.00 65.24      A    C
ATOM   4474  HA1 GLY A 285      81.839  31.680  76.810  1.00  0.00      A    H
ATOM   4475  HA2 GLY A 285      80.875  32.332  78.204  1.00  0.00      A    H
ATOM   4476  C   GLY A 285      82.663  31.330  78.727  1.00 66.49      A    C
ATOM   4477  O   GLY A 285      83.712  31.828  79.217  1.00  0.00      A    O
ATOM   4478  OXT GLY A 285      82.247  30.169  78.986  1.00  0.00      A    O
ENDMDL
END
```

* PDB must have canonical 20 amino acids.

* Remove HETATM waters and ligands

* PDB must not be missing residues

* PDB must use HIS, not HSE or HIE

***Results***

The results will be a pandas dataframe for the trajectory that contains different information depending on what was
specified in the .config file. To learn more about slicing and manipulating pandas data frames go here [https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)
below is an example output for a single frame for the all-atom simulations of 1YTA dimer with two chains/domains A/B

| Syntax      | Description |
| :---        |        ---: |
| frame_num | frame number from trajectory file |
| time | mdtime for frame|
| Q{domain label} | fraction of native contacts for each domain and total structure |
| RMSD{domain label} | root mean square deviation of each domain and total structure |
| G{domain} label | fraction of native contacts with change in entanglement for each domain and between domains. i.e. GAB is G for a thread in A therading a loop in B |
| frame_Maxg('{domain label}', '{domain label}') | maximal total linking number for a given domain pair (wchngent suffix considers only those native contacts with a change in ent from the reference structure) |
| frame_Maxg_dict_resid('{domain label}', '{domain label}') | dictionary containing resid and chainid for the native contact residues that share the frame_Maxg total linking number (wchngent suffix considers only those native contacts with a change in ent from the reference structure) |
| frame_Gr('{domain label}', '{domain label}') | dictionary containing the native contacts that share the frame_Maxg and the residues identified to be at the entanglement site (wchngent suffix considers only those native contacts with a change in ent from the reference structure) |
| frame_S{1\|2} | S: S1/S2 where S1 = -average native contact potential holding loops together, and S2 = Average thread flux through loops. (wchngent suffix considers only those native contacts with a change in ent from the reference structure) (See (Parameter_descriptions)_entanglement_heuristic_filtering_metrics.pdf for more details) |
| freqG{0,1,2,3,4}('{domain label}', '{domain label}') | frequency of a given change in entanglement type observed (see (Parameter_descriptions)_entanglement_heuristic_filtering_metrics.pdf for more details) |


<u>**Example of frame output:**</u>

```python
frame_num                                                                           1
time                                                                               10
QA                                                                           0.436464
QB                                                                           0.317919
Qtotal                                                                       0.350404
RMSDA                                                                         4.66202
RMSDB                                                                         6.59426
RMSDtotal                                                                      5.7778
GAA                                                                          0.019656
GAB                                                                                 0
GBA                                                                                 0
GBB                                                                                 0
frame_MaxgAA                                                                        1
frame_Maxg_dict_residAA             {((<Atom CA [-1]; In TRP -1>), (<Atom CA [-1];...
frame_Maxg_wchngentAA                                                               1
frame_Maxg_wchngent_dict_residAA    {((<Atom CA [-1]; In TRP -1>), (<Atom CA [-1];...
frame_MaxgAB                                                                        0
frame_Maxg_dict_residAB             {((<Atom CA [-1]; In ALA -1>), (<Atom CA [-1];...
frame_Maxg_wchngentAB                                                               0
frame_Maxg_wchngent_dict_residAB                                                   {}
frame_MaxgBA                                                                        0
frame_Maxg_dict_residBA             {((<Atom CA [-1]; In GLU -1>), (<Atom CA [-1];...
frame_Maxg_wchngentBA                                                               0
frame_Maxg_wchngent_dict_residBA                                                   {}
frame_MaxgBB                                                                        0
frame_Maxg_dict_residBB             {((<Atom CA [-1]; In ALA -1>), (<Atom CA [-1];...
frame_Maxg_wchngentBB                                                               0
frame_Maxg_wchngent_dict_residBB                                                   {}
frame_GrAA                          {((<Atom CA [-1]; In TRP -1>), (<Atom CA [-1];...
frame_GrAB                                                                         {}
frame_GrBA                                                                         {}
frame_GrBB                                                                         {}
frame_Gr_wchngentAA                 {((<Atom CA [-1]; In TRP -1>), (<Atom CA [-1];...
frame_Gr_wchngentAB                                                                {}
frame_Gr_wchngentBA                                                                {}
frame_Gr_wchngentBB                                                                {}
frame_SAA                                                                     1.07453
frame_S1AA                                                                   0.427778
frame_S2AA                                                                   0.398108
frame_SAB                                                                           0
frame_S1AB                                                                          0
frame_S2AB                                                                          0
frame_SBA                                                                           0
frame_S1BA                                                                          0
frame_S2BA                                                                          0
frame_SBB                                                                           0
frame_S1BB                                                                          0
frame_S2BB                                                                          0
frame_S_wchngentAA                                                              1.238
frame_S1_wchngentAA                                                          0.492857
frame_S2_wchngentAA                                                          0.398108
frame_S_wchngentAB                                                                  0
frame_S1_wchngentAB                                                                 0
frame_S2_wchngentAB                                                                 0
frame_S_wchngentBA                                                                  0
frame_S1_wchngentBA                                                                 0
frame_S2_wchngentBA                                                                 0
frame_S_wchngentBB                                                                  0
frame_S1_wchngentBB                                                                 0
frame_S2_wchngentBB                                                                 0
freqG0AA                                                                            7
freqG1AA                                                                            0
freqG2AA                                                                            1
freqG3AA                                                                            0
freqG4AA                                                                            0
freqG0AB                                                                            0
freqG1AB                                                                            0
freqG2AB                                                                            0
freqG3AB                                                                            0
freqG4AB                                                                            0
freqG0BA                                                                            0
freqG1BA                                                                            0
freqG2BA                                                                            0
freqG3BA                                                                            0
freqG4BA                                                                            0
freqG0BB                                                                            0
freqG1BB                                                                            0
freqG2BB                                                                            0
freqG3BB                                                                            0
freqG4BB                                                                            0

```

Note: pandas summaries may display incorrect informaiton. Slice data to observe true residues.

## Running analysis on example trajectory:
1) go to the `entanglement_analysis` folder, where contains `command_entanglement_analysis.sh` file

In `entanglement_analysis/setup` folder, we provided all files required for running entanglement analysis:

* python script for analysis: `topology_analysis_v9.3.py`
* config file: `topology_analysis_v9.3.config`
* secondary structure definitions: `2ww4_sec_structure.txt`
* native structure of protein: `2ww4_chain_a_rebuilt_mini.pdb`

In `traj` folder, we provide a trajectory with 100 frames (`md_protein_nopbc_100frames.xtc`) and topology to load trajectory (`newbox.gro`)

2) Execute the following command from `entanglement_analysis` folder :

```bash
python setup/topology_analysis_v9.3.py setup/topology_analysis_v9.3.config 2ww4_ ../traj/md_protein_nopbc_100frames.xtc 0 -1 1 
```

The above command will run an analysis on all frames in the provided trajectory.
The analysis is quite fast, it takes about 1s for for each frame.

The script will create a folder `topology_analysis_9.3` containing subfolder `output` where the resulting pandas data frame file (`*.df`) file saved there.

3) For the gain of non-native entanglement structure, we are interested in the number of contacts gain entanglement only (G0: gain of entanglement without changing chirality and G1: gain of entanglement and change in chirality). 
    To extract the number of contacts gain entanglement, use the following code snippet:

```python
import pandas as pd 

df = pd.read_pickle(f'2ww4__s0_e-1_m1.df')
df_gain = df[['frame_num', 'time', 'freqG0AA', 'freqG1AA']]
df_gain.to_csv('gain_entanglement.dat', sep='\t', float_format="%8f")
```