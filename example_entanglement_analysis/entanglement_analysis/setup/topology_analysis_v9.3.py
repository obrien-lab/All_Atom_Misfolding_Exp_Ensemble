#!/usr/bin/env python3
################################################################################################################
script_title='topology_analysis'
script_version=9.3
script_author='Ian Sitarik'
script_updatelog=f"""Update log for {script_title} version {script_version}

                   Date: 12.28.2020
                   Note: Started covertion of gcalcs to python

                   Date: 12.29.2020
                   Note: Added start, stride, and end frame slicing for dcd
                         added interdomain entanglement analysis

                   Date: 01.10.2021
                   Note: added threshold for discarding terminal residues. This threshold is
                         a fraction of the protein length to discard from each termini
                         when considering native contacts that have changes in entanglemnt.

                   Date: 01.13.2021
                   Note: added entanglement residue identification and output. This is done by finding the
                         MAX[|g(i,j)|] = g(i,j)* for a gvals file output from calc_entanglement_number_v3.1.py
                         then for each termini of g(i,j)* slide a window of length W(m) with middle residue m
                         along the tail 1 step at at time. for each step calculate the partial gp(i,j,W(m)) and
                         outputs a file with the middle residue of the sliding window vs gp(i,j,W(m))

                   Date: 01.31.2021
                   Note: chagned from argument parser to config parser. To many arguments!

                   Date: 02.01.2021
                   Note: added ability to skip intermonomer enanglements and
                         to calculate Q and RMSD for each frame as well

                   Date: 03.14.2021
                   Note: recast script as topology_analsis_v7.1 with an aim at making an all in one script to analyze
                         the following:

                         Q: fraction of native contacts
                         Maxg: Maximal total linking number and its associated loop closing native contacts
                         G: fraction of loop closing native contacts with a change in entanglement
                         ProbGk: discrete probability/frequency distribution of change in entangelment types
                         Gr: set of residues near Maxg entanglement (i.e. residues where terminal tail threads loop)
                         S: scoring function for entanglement stability S1/S2
                            where S1 quantifies the average contact potential forming the loop base
                            and S2 quantifies the ease of unthreading

                   Note: made supplying original all-atom PDB used to make a CG model mandetory so
                         native contact potentials can be estimated.

                   Date: 03.15.2021
                   Note: added multidomain/chain analysis for Q and entanglement
                         Q will yeild one result per dom/chain and one result for each pair of dom/chain and a total
                            example for two chains A and B there will be 3 Q analysis results
                            Q(A->A) fract nc in A
                            Q(B->B) fract nc in B
                            Q(A->B) = Q(B->A) fract nc between A & B
                            Q(A->A,B->B,A->B) total fraction nc

                         G will yeild one result per dom/chain and two results for each pair of dom/chain and a total
                            example for two chains A and B there will be 4 G analysis results
                            G(A->A) ent of thread in A with loop in A
                            G(B->B) ent of thread in B with loop in B
                            G(A->B) ent of thread in A with loop in B
                            G(B->A) ent of thread in B with loop in A
                            G(A->A, B->B, A->B, B->A) total ent

                    Date: 03.28.2021
                    Note: Added ability to turn fraction_native_contacts & entanglement_analysis on and off
                          in the config file
                    Note: added RMSD calc

                    Date: 03.29.2021
                    Note: changed entanglement residue output to be a dictionary
                          where each native contact reports its own set of entangled residues

                    Date: 04.05.2021
                    Note: added ability for user to define mutliple residue masks for Q
                          added calculations for residue specific SASA and structural overlap

                    Date: 04.10.2021
                    Note: cleaned up some data structures and added missing residue functionality

                    Date: 04.20.2021
                    Note: added the following new features for v9.1
                        - print_frame_summary option in config file to allow for the binary toggling of
                        frame summary output for debugging
                        - rmsd_mask option in config file to allow for monitoring of RMSD for specific residues
                  """

################################################################################################################

import os
import sys
import numpy as np
import time
import datetime
#import pickle
#import argparse
import re
import itertools
import multiprocessing
import warnings
from MDAnalysis import *
from scipy.spatial.distance import pdist, squareform
from itertools import product
import configparser
import MDAnalysis.analysis.rms
import parmed as pmd
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

##################################################################################################################
### START argument parse ###
##################################################################################################################
if len(sys.argv) != 7:
    print(f'[1] = path to config file')
    print(f'[2] = outfile_basename')
    print(f'[3] = in_path')
    print(f'[4] = start_frame')
    print(f'[5] = end_frame')
    print(f'[6] = frame_stride')
    quit()

config = configparser.ConfigParser()
config.read(sys.argv[1])
default = config['DEFAULT']

in_path=sys.argv[3]
print(f'in_path: {in_path}')
if in_path == None: print(f'\n{script_updatelog}\n'); sys.exit()
else: in_paths=[x.strip('\n').split(', ') for x in os.popen(f'ls -v {in_path}').readlines()]
for ipath in in_paths: print(f'{ipath}')

out_path=default['out_path']
print(f'out_path: {out_path}')
if out_path == None: print(f'\n{script_updatelog}\n'); sys.exit()

psf=default['psf']
print(f'psf: {psf}')
if psf == None: print(f'\n{script_updatelog}\n'); sys.exit()

bt_contact_pot_file=default['bt_contact_pot_file']
print(f'bt_contact_pot_file: {bt_contact_pot_file}')
if bt_contact_pot_file == None: print(f'\n{script_updatelog}\n'); sys.exit()

print_updatelog=default['print_updatelog']
print(f'print_updatelog: {print_updatelog}')
if print_updatelog != None: print(f'\n{script_updatelog}\n')

global print_framesummary
print_framesummary=default.getboolean('print_framesummary')
print(f'print_framesummary: {print_framesummary}')
if print_framesummary == None: print(f'\n{script_updatelog}\n')

window_size=default['window_size']
print(f'window_size: {window_size}')
if window_size == None: print(f'\n{script_updatelog}\n'); pass
else: window_size=int(window_size)

outfile_basename=sys.argv[2]
print(f'outfile_basename: {outfile_basename}')
if outfile_basename == None: print(f'\n{script_updatelog}\n'); sys.exit()

start_frame=sys.argv[4]
print(f'start_frame: {start_frame}')
if start_frame == None:
    print(f'\n{script_updatelog}\n')
    start_frame = 0
else: start_frame=int(start_frame)

end_frame=sys.argv[5]
print(f'end_frame: {end_frame}')
if end_frame == None:
    print(f'\n{script_updatelog}\n')
    end_frame = 999999999999
else: end_frame=int(end_frame)

frame_stride=sys.argv[6]
print(f'frame_stride: {frame_stride}')
if frame_stride == None:
    print(f'\n{script_updatelog}\n')
    frame_stride = 1
else: frame_stride=int(frame_stride)

termini_threshold=default['termini_threshold']
print(f'termini_threshold: {termini_threshold}')
if termini_threshold == None:
    print(f'\n{script_updatelog}\n')
    termini_threshold = [5,5]
else: termini_threshold=[int(x) for x in termini_threshold.split(',')]

model_res=default['model_res']
print(f'model_res: {model_res}')
if model_res == None: print(f'\n{script_updatelog}\n'); sys.exit()

cg_ref_crd=default['cg_ref_crd']
print(f'cg_ref_crd: {cg_ref_crd}')
if cg_ref_crd == None: print(f'\n{script_updatelog}\n')

orig_aa_pdb=default['orig_aa_pdb']
print(f'orig_aa_pdb: {orig_aa_pdb}')
if orig_aa_pdb == None: print(f'\n{script_updatelog}\n'); sys.exit()

global resid_masks
resid_masks = default['resid_masks']
if resid_masks == None: print(f'\n{script_updatelog}\n');
elif resid_masks == 'nan':
    resid_masks = None
else:
    try:
        resid_masks={k:[v.split('-')[1],[int(x) for x in v.split('-')[0].split(',')]] for k,v in [x.split(':') for x in default['resid_masks'].split(';')]}
    except:
        resid_masks = None
print(f'resid_masks: {resid_masks}')

global rmsd_masks
rmsd_masks = default['rmsd_masks']
if rmsd_masks == 'nan': rmsd_masks = None
print(f'rmsd_masks: {rmsd_masks}')

ref_only=default.getboolean('ref_only')
print(f'ref_only: {ref_only}')
if ref_only == None: print(f'\n{script_updatelog}\n'); sys.exit()

global ent_res_method
ent_res_method = default.getint('ent_res_method')
if ent_res_method == None: ent_res_method = 2
print(f'ent_res_method: {ent_res_method}')

sections = config.sections()
if 'ANALYSIS' in sections:
    additional_analysis_flags = config['ANALYSIS']

    if 'fraction_native_contact_analysis' in additional_analysis_flags:
        fraction_native_contact_analysis = additional_analysis_flags.getboolean('fraction_native_contact_analysis')
        print(f'fraction_native_contact_analysis: {fraction_native_contact_analysis}')
    else: fraction_native_contact_analysis = None

    if 'fraction_native_contacts_wchngent_analysis' in additional_analysis_flags:
        fraction_native_contacts_wchngent_analysis = additional_analysis_flags.getboolean('fraction_native_contacts_wchngent_analysis')
        print(f'fraction_native_contacts_wchngent_analysis: {fraction_native_contacts_wchngent_analysis}')
    else: fraction_native_contacts_wchngent_analysis = None

    if 'rmsd_analysis' in additional_analysis_flags:
        rmsd_analysis = additional_analysis_flags.getboolean('rmsd_analysis')
        print(f'rmsd_analysis: {rmsd_analysis}')
    else:
        rmsd_analysis = None

    if 'struct_overlap_analysis' in additional_analysis_flags:
        struct_overlap_analysis = additional_analysis_flags.getboolean('struct_overlap_analysis')
        print(f'struct_overlap_analysis: {struct_overlap_analysis}')
    else:
        struct_overlap_analysis = None

    if 'entanglement_analysis' in additional_analysis_flags:
        entanglement_analysis = additional_analysis_flags.getboolean('entanglement_analysis')
        print(f'entanglement_analysis: {entanglement_analysis}')
    else:
        entanglement_analysis = None

else:
    print(f'No Analysis was specified. Defaulting to ref_only')
    ref_only = True

##################################################################################################################
### START initial loading of structure files and qualtiy control ###
##################################################################################################################

#load ref_structure
ref_structure = pmd.formats.pdb.PDBFile(orig_aa_pdb).parse(orig_aa_pdb)[:,:,'CA']
print(f'ref_structure: {ref_structure} loaded from {orig_aa_pdb}')

#load ref_cg_coordinates onto ref pmd structures
if model_res == 'cg':
    temp_coor = pmd.charmm.CharmmCrdFile(cg_ref_crd).coordinates[0]
    for i,(x,y,z) in enumerate(temp_coor):
        ref_structure.atoms[i].xx = x.astype(np.float32)
        ref_structure.atoms[i].xy = y.astype(np.float32)
        ref_structure.atoms[i].xz = z.astype(np.float32)

#domain and secondary structure inputs
#user specifies an Amber like atom selection for each domain
#in dictionary format with the following syntax
# {'domain_name': #-#-'chain/segid'-'atomname', ...}
dom_def=default['dom_def']
try:
    dom_def = {k:v.split('-') for k,v in [x.split(':') for x in dom_def.split(',')]}
except:
    #default dom_def pulled from ref pdb
    dom_def = {}
    chain_list = set()
    for residue in ref_structure.residues:
        chain_list.add(residue.chain)
    for chain in chain_list:
        chain_res = [res.number for res in ref_structure[chain,:,'CA'].residues]
        start_res = min(chain_res)
        end_res = max(chain_res)

        dom_def[chain] = [start_res,end_res,chain,'CA']


print(f'dom_def: {dom_def}', type(dom_def))
if dom_def == None: print(f'\n{script_updatelog}\n'); sys.exit()

#list of unique domains in sec_elems_file
udom_labels = np.unique(np.asarray(list(dom_def.keys())))
global udom_pairs
udom_pairs = list(itertools.product(udom_labels,udom_labels))
print(f'udom_pairs: {udom_pairs}')

#if Q get and parse secondary structure file
if fraction_native_contact_analysis:

    #make a matrix containing the secondary structure elements
    sec_elems_file_path = default['sec_elems_file_path']
    sec_elems_file = np.asarray([x.strip('\n').split() for x  in open(sec_elems_file_path).readlines()])
    print(sec_elems_file)

    try:
        sec_elems = [[int(start),int(stop),domdef,chaindef] for _,start,stop,domdef,chaindef in sec_elems_file]
    except:
        print(f'secondary structure file formate missing columns. check the readme for updates and format')
        sys.exit()
    sec_struct_udom_labels = np.unique(sec_elems_file[:,-2])

    #print(sec_elems, len(sec_elems))
    if len(sec_struct_udom_labels) != len(dom_def):
        print(f'ERROR: the number of unique domains in {sec_elems_file_path} {len(sec_elems)} != the number of domains in {dom_def} {len(dom_def)}')
        quit()


#parse rmsd_masks
global rmsd_masks_dom_dict
rmsd_masks_dom_dict = None
if rmsd_masks != None:
    rmsd_masks_dom_dict = {}
    print(f'rmsd_masks: {rmsd_masks}')
    if rmsd_masks == '2structs':
        print(f'rmsd_masks will be set to residues in 2structs as defined by sec_elems_file_path')

        #make a matrix containing the secondary structure elements
        sec_elems_file_path = default['sec_elems_file_path']
        sec_elems_file = np.asarray([x.strip('\n').split() for x  in open(sec_elems_file_path).readlines()])

        try:
            sec_elems = [[int(start),int(stop),domdef,chaindef] for _,start,stop,domdef,chaindef in sec_elems_file]
        except:
            print(f'secondary structure file formate missing columns. check the readme for updates and format')
            sys.exit()
        else:
            udoms = np.unique([d[2] for d in sec_elems])
            for udom in udoms:
                rmsd_masks_dom_dict[udom] = np.hstack([np.arange(d[0],d[1]+1) for d in sec_elems if d[2] == udom])
    else:
        try:
            rmsd_masks_dom_dict={k:np.asarray([int(x) for x in v.split('-')[0].split(',')]) for k,v in [x.split(':') for x in rmsd_masks.split(';')]}

        except:
            rmsd_masks = None
            rmsd_masks_dom_dict = None

    #check that rmsd_masks are in dom_def ranges
    temp_rmsd_masks_dom_dict = {}
    for dom,rmsd_mask in rmsd_masks_dom_dict.items():
        dom_range = np.arange(int(dom_def[dom][0]),int(dom_def[dom][1]))
        temp_rmsd_masks_dom_dict[dom] = [d for d in rmsd_mask if d in dom_range]

    rmsd_masks_dom_dict = temp_rmsd_masks_dom_dict
    del temp_rmsd_masks_dom_dict

print(f'rmsd_masks_dom_dict: {rmsd_masks_dom_dict}')

#load in reference strucutre, reorder coordinates, and
#build list of residues not in sec_elems but still in each domain
#build dictionary for converting coordinate index to resid
global anti_res_in_sec_elems_dom_dict, corid2resid_dom_dict, resid2corid_dom_dict
global dom_ref_structures, anti_corid_in_sec_elems_dom_dict, total_dom_cooridx_dict
anti_corid_in_sec_elems_dom_dict = {}
anti_res_in_sec_elems_dom_dict = {}
corid2resid_dom_dict = {}
resid2corid_dom_dict = {}
dom_ref_structures = {}
dom_ref_structures_coor = {}
total_dom_cooridx_dict ={}

for k,v in dom_def.items():
    res_in_sec_elems_list=[]
    start_res = int(v[0])
    end_res = int(v[1])
    chain = v[2]
    atomname = v[3]
    corid2resid_dom_dict[k] = {}
    resid2corid_dom_dict[k] = {}
    print(f'building dom {k} structure with start_res {start_res} end_res {end_res} chain {chain} atomname {atomname}')

    #build dom_ref_struct dictionary
    dom_ref_structures[k] = ref_structure[chain,:,atomname]


    #find if residues are missing
    resids = [res.number for res in dom_ref_structures[k].residues]
    #print(resids)
    user_resids = np.arange(start_res, end_res + 1)
    #print(user_resids)
    missing_res = [res for res in user_resids if res not in resids]
    print(f'missing_res: {missing_res}')

    #make sure coordinates are in the proper order
    cor_res_list = [[c,r,n] for c,r,n in zip(dom_ref_structures[k].coordinates, dom_ref_structures[k].residues, resids)]
    cor_res_list.sort(key=lambda x: x[2])
    cor_res_list = [x for x in cor_res_list if x[2] in range(start_res,end_res+1)]
    cor_res_list = [[i]+cor_res for i,cor_res in enumerate(cor_res_list)]
    for cor_res in cor_res_list: print(cor_res)


    dom_ref_structures[k] = dom_ref_structures[k][chain,0:len(cor_res_list),atomname]

    for new_coorid,coor,res,resnum in cor_res_list:
        #print(new_coorid,coor,res,resnum)

        dom_ref_structures[k].atoms[new_coorid].xx = coor[0].astype(np.float32)
        dom_ref_structures[k].atoms[new_coorid].xy = coor[1].astype(np.float32)
        dom_ref_structures[k].atoms[new_coorid].xz = coor[2].astype(np.float32)
        dom_ref_structures[k].residues[new_coorid] = res

        corid2resid_dom_dict[k][new_coorid] = res
        resid2corid_dom_dict[k][res] = new_coorid

    #cor_res_list = [[c,r] for c,r in zip(dom_ref_structures[k].coordinates, dom_ref_structures[k].residues)]
    for cor_res in cor_res_list: print(cor_res)

    #print(dom_ref_structures[k].residues)
    #print(dom_ref_structures[k].coordinates[:10])
    dom_ref_structures_coor[k] = dom_ref_structures[k].coordinates.astype(np.float32)

    # if Q requested get sec struct info and anti_corid_in_sec_elems_array
    resids = [x for x in dom_ref_structures[k].residues]
    if fraction_native_contact_analysis:
        #res not in 2structs finder

        for elem in sec_elems:
            print(elem,k,chain)
            if elem[2] == k:
                if elem[3] == chain:
                    ress=np.arange(elem[0],elem[1]+1)
                    res_in_sec_elems_list.append(ress)
        print(res_in_sec_elems_list)
        anti_res_in_sec_elems_dom_dict[k] = [x for x in resids if x.number not in list(np.hstack(res_in_sec_elems_list))]
        corids = np.asarray([resid2corid_dom_dict[k][x] for x in anti_res_in_sec_elems_dom_dict[k]])
        anti_corid_in_sec_elems_dom_dict[k] = corids


#make total structure and its corid2resid dicts if there is more than 1 domain
if len(dom_ref_structures.keys()) > 1:
    for i,(dom,struct) in enumerate(dom_ref_structures.items()):
        #print(i,dom,struct)
        if i == 0:

            total_struct = struct.copy(pmd.structure.Structure)
            #print(total_struct.residues)
            resids = [x for x in struct.residues]
            if fraction_native_contact_analysis:
                anti_corids_list = [np.asarray([resid2corid_dom_dict[dom][x] for x in anti_res_in_sec_elems_dom_dict[dom]])]
            total_dom_cooridx_dict[dom] = np.arange(len(struct.residues))
            last_res = len(struct.residues)

        elif i > 0:
            total_struct += struct.copy(pmd.structure.Structure)
            #print(total_struct.residues)
            resids += [x for x in struct.residues]
            if fraction_native_contact_analysis:
                anti_corids_list += [np.asarray([resid2corid_dom_dict[dom][x] for x in anti_res_in_sec_elems_dom_dict[dom]]) + last_res - 1]
            total_dom_cooridx_dict[dom] = np.arange(len(struct.residues)) + last_res
            last_res = last_res + len(struct.residues)

    total_dom_cooridx_dict['total'] = np.hstack([d for d in total_dom_cooridx_dict.values()])
    #print(len(total_dom_cooridx_dict['total']), len(total_struct.residues))
    if fraction_native_contact_analysis:
        anti_corid_in_sec_elems_dom_dict['total']=np.hstack(anti_corids_list)
    corids = np.arange(len(resids))
    corid2resid_dom_dict['total'] = {c:r for c,r in zip(corids,resids)}
    resid2corid_dom_dict['total'] = {r:c for r,c in zip(resids,corids)}
    dom_ref_structures['total'] = total_struct
    dom_ref_structures_coor['total'] = total_struct.coordinates


#make resnum2corid_dom_dict
global resnum2corid_dom_dict
resnum2corid_dom_dict = {}
global corid2resnum_dom_dict
corid2resnum_dom_dict = {}

for dom,resid2corid in resid2corid_dom_dict.items():
    resnum2corid_dom_dict[dom] = {resid.number:corid for resid,corid in resid2corid.items()}
    corid2resnum_dom_dict[dom] = {corid:resid.number for resid,corid in resid2corid.items()}


#contains resnum to corid instead of full residue objecto to corid
if resid_masks != None:
    global corid_masks
    corid_masks = {}
    for dom,resid2corid in resid2corid_dom_dict.items():

        if dom == 'total':

            mask_labels = [''.join([k,v[0]]) for k,v in resid_masks.items()]
            total_mask_labels_str = ''.join(mask_labels)

            total_corid_mask = []
            for mask_dom,mask_resid in resid_masks.items():
                mask_resid = mask_resid[1]

                for res in mask_resid:
                    for resid,corid in resid2corid.items():
                        if resid.chain == mask_dom:
                            if resid.number == res:
                                total_corid_mask += [corid]
                                break

            corid_masks[dom] = [total_mask_labels_str] + [total_corid_mask]

        else:
            dom_resid_mask = resid_masks[dom]
            corid_masks[dom] = [dom_resid_mask[0]] + [[resnum2corid_dom_dict[dom][r] for r in dom_resid_mask[1]]]


#convert rmsd_masks_dom_dict to coorid for each domain
global rmsd_masks_dom_coorid_dict
rmsd_masks_dom_coorid_dict = None
if rmsd_masks != None:
    rmsd_masks_dom_coorid_dict = {}
    for i,(dom,rmsd_mask_resids) in enumerate(rmsd_masks_dom_dict.items()):
        print(i,dom,rmsd_mask_resids)

        rmsd_masks_dom_coorid_dict[dom] = [resnum2corid_dom_dict[dom][x] for x in rmsd_mask_resids]
        print(rmsd_masks_dom_coorid_dict[dom])


        if len(rmsd_masks_dom_dict) > 1:
            if i == 0:
                rmsd_masks_dom_coorid_dict['total'] = [resnum2corid_dom_dict['total'][x] for x in rmsd_mask_resids]
            if i > 0:
                rmsd_masks_dom_coorid_dict['total'] += [resnum2corid_dom_dict['total'][x] for x in rmsd_mask_resids]

#print(f'total_dom_cooridx_dict: {total_dom_cooridx_dict}\n')
#print(f'corid2resid_dom_dict: {corid2resid_dom_dict}\n')
#print(f'resid2corid_dom_dict: {resid2corid_dom_dict}\n')
#print(f'anti_res_in_sec_elems_dom_dict: {anti_res_in_sec_elems_dom_dict}\n')
#print(f'anti_corid_in_sec_elems_dom_dict: {anti_corid_in_sec_elems_dom_dict}\n')
#print(f'dom_ref_structures: {dom_ref_structures}\n')
##################################################################################################################
### END initial loading of structure files and qualtiy control ###
##################################################################################################################

### START dir declaration ###

if os.path.exists(f'{out_path}/'):
    print(f'{out_path}/ does exists and will be used')
    pass
else:
    os.makedirs(f'{out_path}/')

if os.path.exists(f'{out_path}{script_title}_{script_version}/'):
    print(f'{out_path}{script_title}_{script_version}/ does exists and will be used')
    pass
else:
    os.makedirs(f'{out_path}{script_title}_{script_version}/')

if os.path.exists(f'{out_path}{script_title}_{script_version}/logs/'):
    print(f'{out_path}{script_title}_{script_version}/logs/ does exists and will be used')
    pass
else:
    os.makedirs(f'{out_path}{script_title}_{script_version}/logs/')

if os.path.exists(f'{out_path}{script_title}_{script_version}/output/'):
    print(f'{out_path}{script_title}_{script_version}/output/ does exists and will be used')
    pass
else:
    os.makedirs(f'{out_path}{script_title}_{script_version}/output/')

if os.path.exists(f'{out_path}{script_title}_{script_version}/gn_gc_files/'):
    print(f'{out_path}{script_title}_{script_version}/gn_gc_files/ does exists and will be used')
    pass
else:
    os.makedirs(f'{out_path}{script_title}_{script_version}/gn_gc_files/')
### END dir declaration ###

### START preference setting ###

start_time=time.time() #time since epoch
print('time since epoch = '+str(start_time))

now=datetime.datetime.now() #time now at this moment
print('time now at this moment = '+str(now))
np.set_printoptions(threshold=sys.maxsize,linewidth=200)  #print total numpy matrix
np.set_printoptions(linewidth=200)  #print total numpy matrix
np.seterr(divide='ignore')

### END preference setting ###

######################################################################################################################
# USER DEFINED FUNCTIONS                                                                                             #
######################################################################################################################

def gen_nc_gdict(struct_dict, coor_cmap_dict, termini_threshold, **kwargs):
    dom_nc_gdict = {}
    dom_gn_dict = {}
    dom_gc_dict = {}

    Nterm_thresh = termini_threshold[0]
    Cterm_thresh = termini_threshold[1]
    #print(f'\ngen_nc_gdict analysis')
    #print(f'udom_pairs: {udom_pairs}')

    for dom_pair in udom_pairs:
        #print(dom_pair)
        thread_dom = dom_pair[0]
        loop_dom = dom_pair[1]

        #get coordinates, native contact list and length of considered domains
        if thread_dom == loop_dom:
            coor_cmap = coor_cmap_dict[thread_dom][0]
            nc_indexs = np.stack(np.nonzero(coor_cmap)).transpose()
            #print(np.nonzero(coor_cmap))
            #print(nc_indexs.shape)
            #print(nc_indexs)

            l = len(struct_dict[thread_dom].residues)
            coor = struct_dict[thread_dom].coordinates
            #print(coor_cmap.shape, l, coor.shape)

        else:
            coor_cmap = coor_cmap_dict['total'][0]
            nc_indexs = np.stack(np.nonzero(coor_cmap)).transpose()
            loop_dom_coridxs = total_dom_cooridx_dict[loop_dom]
            nc_indexs = np.stack([nc for nc in nc_indexs if (nc[0] in loop_dom_coridxs) & (nc[1] in loop_dom_coridxs)])

            l = len(struct_dict['total'].residues)
            coor = struct_dict['total'].coordinates
            #print(coor_cmap.shape, l, coor.shape)

        nc_gdict = {}

        #make R and dR waves of length N-1
        range_l = np.arange(0, l-1)
        range_next_l = np.arange(1,l)

        R = 0.5*(coor[range_l] + coor[range_next_l])
        dR = coor[range_next_l] - coor[range_l]

        #make dRcross matrix
        pair_array = np.asarray(list(itertools.product(dR,dR)))

        x = pair_array[:,0,:]
        y = pair_array[:,1,:]

        dR_cross = np.cross(x,y)

        #make Rnorm matrix
        pair_array = np.asarray(list(itertools.product(R,R)))

        diff = pair_array[:,0,:] - pair_array[:,1,:]
        Runit = diff / np.linalg.norm(diff, axis=1)[:,None]**3
        Runit = Runit.astype(np.float32)

        #make final dot matrix
        dot_matrix = [np.dot(x,y) for x,y in zip(Runit,dR_cross)]
        dot_matrix = np.asarray(dot_matrix)
        dot_matrix = dot_matrix.reshape((l-1,l-1))
        print(type(dot_matrix[0,1]))
        print(type(dot_matrix))

        #calc gn and gc for each native contact
        #and make output dictionary
        gn_outdata = []
        gc_outdata = []
        for i,j in nc_indexs:

            if (i >= 0) and (j < l-1):
                #print(f'\nnc: {i} {j}')
                loop_range = np.arange(i,j)
                #nterm_range = np.arange(5,i-4)
                #cterm_range = np.arange(j+4,l-6)

                if thread_dom == loop_dom:
                    #print(Nterm_thresh,(i-4), (j+4),(l-Cterm_thresh-1),l, l-Cterm_thresh)
                    nterm_range = np.arange(Nterm_thresh,i-4)
                    cterm_range = np.arange(j+4,l-Cterm_thresh-1)

                else:
                    nterm_range = np.arange(5,2)
                    cterm_range = total_dom_cooridx_dict[thread_dom][:-1]

                gn_pairs_array = np.asarray(list(itertools.product(nterm_range,loop_range)))
                #print('gn_pairs_array:\n',gn_pairs_array[:10], '\n',gn_pairs_array[-10:])
                gc_pairs_array = np.asarray(list(itertools.product(loop_range, cterm_range)))
                #print('gc_pairs_array:\n',gc_pairs_array[:10], '\n',gc_pairs_array[-10:])

                if len(gn_pairs_array) != 0:
                    #gn_vals = np.zeros((dot_matrix.shape),dtype=bool)
                    #gn_vals[gn_pairs_array[:,0],gn_pairs_array[:,1]] = True
                    #gn_vals = dot_matrix[gn_vals]
                    gn_vals = dot_matrix[gn_pairs_array[:,0],gn_pairs_array[:,1]]
                    gn_val = gn_vals.sum()/(4.0*np.pi)

                    num_gn_vals = gn_vals.shape[0]
                    start_end_loop_res_array = np.full((num_gn_vals,2),[i,j])
                    temp_outdata = np.hstack((start_end_loop_res_array,np.flip(gn_pairs_array, axis = 1), gn_vals[:,None]))

                    if abs(round(gn_val)) > 0:
                        gn_outdata.append(temp_outdata)

                else:
                    gn_val = 0

                if len(gc_pairs_array) != 0:
                    #gc_vals = np.zeros((dot_matrix.shape),dtype=bool)
                    #gc_vals[gc_pairs_array[:,0],gc_pairs_array[:,1]] = True
                    #gc_vals = dot_matrix[gc_vals]
                    #print(gc_vals.shape,gc_vals)
                    gc_vals = dot_matrix[gc_pairs_array[:,0],gc_pairs_array[:,1]]
                    gc_val = gc_vals.sum()/(4.0*np.pi)
                    #print(gc_val)

                    num_gc_vals = gc_vals.shape[0]
                    start_end_loop_res_array = np.full((num_gc_vals,2),[i,j])
                    temp_outdata = np.hstack((start_end_loop_res_array, gc_pairs_array, gc_vals[:,None]))

                    if abs(round(gc_val)) > 0:
                        #print(temp_outdata)
                        gc_outdata.append(temp_outdata)

                else:
                    gc_val = 0

                total_link = round(gn_val) + round(gc_val)
                #print((i,j), gn_val, gn_pairs_array.shape, gc_val, gc_pairs_array.shape, total_link)
                #print((i,j), gn_val, gc_val, total_link)
                nc_gdict[(i,j)] = total_link


        try:
            gn_outdata = np.vstack(gn_outdata)
        except:
            gn_outdata = np.asarray([])

        try:
            gc_outdata = np.vstack(gc_outdata)
        except:
            gc_outdata = np.asarray([])

        dom_nc_gdict[dom_pair] = nc_gdict
        dom_gn_dict[dom_pair] = gn_outdata
        dom_gc_dict[dom_pair] = gc_outdata

    return dom_nc_gdict, dom_gn_dict, dom_gc_dict

def gen_chng_ent_dict(ref_nc_gdict, frame_nc_gdict):
    chng_ent_dom_pair_dict = {}

    #print('\n',ref_nc_gdict, '\n', ref_nc_gdict.keys(), '\n')
    for dom_pair in frame_nc_gdict.keys():
        chng_ent_dict={}
        if print_framesummary: print(f'\nAnalysing changes in entanglement of dom_pair {dom_pair}')
        frame_ncs = frame_nc_gdict[dom_pair]
        for nc in frame_ncs:

            ref_g = ref_nc_gdict[dom_pair][nc]
            frame_g = frame_nc_gdict[dom_pair][nc]

            if ref_g != frame_g:
                if (abs(frame_g) > abs(ref_g)) and (frame_g*ref_g >= 0):
                    chng_ent_dict[nc] = [0,frame_g]
                if (abs(frame_g) > abs(ref_g)) and (frame_g*ref_g < 0):
                    chng_ent_dict[nc] = [1,frame_g]
                if (abs(frame_g) < abs(ref_g)) and (frame_g*ref_g >= 0):
                    chng_ent_dict[nc] = [2,frame_g]
                if (abs(frame_g) < abs(ref_g)) and (frame_g*ref_g < 0):
                    chng_ent_dict[nc] = [3,frame_g]
                if (abs(frame_g) == abs(ref_g)) and (frame_g*ref_g < 0):
                    chng_ent_dict[nc] = [4,frame_g]

        chng_ent_dom_pair_dict[dom_pair] = chng_ent_dict
    return chng_ent_dom_pair_dict


def cmap(dom_structures, ref = True, restricted = True, cut_off = 8.0, bb_buffer = 4, **kwargs):
    if print_framesummary: print(f'\nCMAP generator')
    if print_framesummary: print(f'ref: {ref}\nrestricted: {restricted}\ncut_off: {cut_off}\nbb_buffer: {bb_buffer}')
    #make cmap for each domain and total structure
    cmap_return_dict = {}
    for dom,struct in dom_structures.items():
        max_dom_coor = struct.coordinates.shape[0]
        cor = struct.coordinates

        distance_map=squareform(pdist(cor,'euclidean'))

        if ref == True:
            distance_map[distance_map>cut_off]=0

            if restricted == True:
                #apply mask
                if resid_masks != None:

                    distance_map=np.triu(distance_map,k=bb_buffer)

                    if len(corid_masks) == 0:
                        print(f'ERROR: len(corid_masks) == 0')
                        quit()


                    dom_corid_mask = corid_masks[dom][-1]
                    anti_dom_corid_mask = [x for x in np.arange(0,max_dom_coor) if x not in dom_corid_mask]

                    distance_map[anti_dom_corid_mask,:] = 0
                    distance_map[:,anti_dom_corid_mask] = 0

                #if no mask specified use sec elems file
                elif resid_masks == None:
                    distance_map=np.triu(distance_map,k=bb_buffer)
                    anti_corid_in_sec_elems_list = anti_corid_in_sec_elems_dom_dict[dom]
                    distance_map[anti_corid_in_sec_elems_list,:]=0
                    distance_map[:,anti_corid_in_sec_elems_list]=0

            elif restricted == False:
                distance_map=np.triu(distance_map,k=bb_buffer)


            contact_map=(distance_map>0).astype(int)

        elif ref == False:

            if restricted == False:
                ref_distance_map = kwargs['ref_cmap_dict'][dom][1]
                ref_contact_map = kwargs['ref_cmap_dict'][dom][0]
                distance_map[distance_map>cut_off]=0
                distance_map=np.triu(distance_map,k=bb_buffer)

                contact_map=(distance_map>0).astype(int)
                contact_map = contact_map*ref_contact_map

            if restricted == True:
                ref_distance_map_restricted = kwargs['ref_cmap_restricted_dict'][dom][1]
                ref_contact_map_restricted = kwargs['ref_cmap_restricted_dict'][dom][0]
                ref_cmap_num_contacts = ref_contact_map_restricted.sum()
                distance_map[distance_map>(ref_distance_map_restricted*1.2)]=0

                #apply mask
                if resid_masks != None:
                    distance_map=np.triu(distance_map,k=bb_buffer)

                    if len(corid_masks) == 0:
                        print(f'ERROR: len(corid_masks) == 0')
                        quit()


                    dom_corid_mask = corid_masks[dom][-1]
                    anti_dom_corid_mask = [x for x in np.arange(0,max_dom_coor) if x not in dom_corid_mask]

                    distance_map[anti_dom_corid_mask,:] = 0
                    distance_map[:,anti_dom_corid_mask] = 0

                #if no mask specified use sec elems file
                elif resid_masks == None:
                    anti_corid_in_sec_elems_list = anti_corid_in_sec_elems_dom_dict[dom]
                    distance_map=np.triu(distance_map,k=bb_buffer)
                    distance_map[anti_corid_in_sec_elems_list,:]=0
                    distance_map[:,anti_corid_in_sec_elems_list]=0

                contact_map=(distance_map>0).astype(int)
                contact_map = contact_map*ref_contact_map_restricted

        contact_num=contact_map.sum()

        if print_framesummary: print(f'Total number of contacts in dom {dom} is {contact_num}')
        cmap_return_dict[dom] = [contact_map, distance_map, contact_num]

    return cmap_return_dict

#parse bt potential file in share_scripts
def bt_parse(bt_file):
    print(f'Parsing bt potential into callable array')
    global aa_names_to_bt_array_idx, bt_array

    data = [x.strip('\n').split() for x in open(bt_file,'r').readlines()]
    aa_names = data[1][1:]

    aa_names_to_bt_array_idx = {aa_name.upper():idx for idx,aa_name in enumerate(aa_names)}
    print(aa_names_to_bt_array_idx)

    data = list(itertools.chain.from_iterable(data[2:]))
    data = np.asarray(data).astype(float)

    bt_array = np.zeros((20,20))
    bt_array[np.tril_indices_from(bt_array,k=0)] = data
    bt_array = bt_array + bt_array.T - np.diag(np.diag(bt_array))
    print(bt_array)

    return bt_array, aa_names_to_bt_array_idx

#make cmap with bt contact potentials
def bt_pot_cmap(bt_array, aa_names_to_bt_array_idx, ref_cmap_dict, res_pair_struct_dict):

    #print(ref_cmap_dict)
    bt_pot_cmap_dict = {}
    #print(res_pair_struct_dict)
    for dom,ref_cmap_objs in ref_cmap_dict.items():

        ref_cmap = ref_cmap_objs[0]
        res_pair_struct_array = res_pair_struct_dict[dom]
        #print(ref_cmap.shape, res_pair_struct_array.shape)

        bt_pot_cmap_array = np.full_like(ref_cmap, 0).astype(float)
        contact_idxs = np.nonzero(ref_cmap)

        res_pairs_in_contact = res_pair_struct_array[contact_idxs]
        resname_pairs_in_contact = np.asarray([[i.residue.name, j.residue.name] for i,j in res_pairs_in_contact])
        resname_pairs_bt_pot = np.asarray([bt_array[aa_names_to_bt_array_idx[i], aa_names_to_bt_array_idx[j]] for i,j in resname_pairs_in_contact])

        bt_pot_cmap_array[contact_idxs] = resname_pairs_bt_pot
        bt_pot_cmap_dict[dom] = bt_pot_cmap_array

    return bt_pot_cmap_dict

#find Gr (residues near entanglemetn site)
def Gr(dom_pair_Maxg_dict,dom_pair_Maxg_nc_dict, dom_pair_gn_vals, dom_pair_gc_vals):

    gmax_res_set_dom_pair_dict = {}
    gmax_res_list_dom_pair_dict = {}

    if print_framesummary: print(f'\nGr analysis')
    for dom_pair,Maxg_dict in dom_pair_Maxg_nc_dict.items():
        #print('\n',dom_pair, Maxg_dict)

        if dom_pair_Maxg_dict[dom_pair] == 0:
            gmax_res_set_dom_pair_dict[dom_pair] = set()
            gmax_res_list_dom_pair_dict[dom_pair] = list()
            continue

        thread_dom = dom_pair[0]
        loop_dom = dom_pair[1]
        gn_vals = dom_pair_gn_vals[dom_pair]
        gc_vals = dom_pair_gc_vals[dom_pair]

        if gn_vals.shape[0] == 0:
            gn_vals = gn_vals.reshape((0,5))
        if gc_vals.shape[0] == 0:
            gc_vals = gc_vals.reshape((0,5))

        gmax_res_list = []
        gmax_res_set = set()
        for Maxg_nc in Maxg_dict.keys():
            #print('\n',Maxg_nc, gn_vals.shape, gc_vals.shape)

            nterm_data = gn_vals[np.logical_and(gn_vals[:,0] == Maxg_nc[0], gn_vals[:,1] == Maxg_nc[1])]
            cterm_data = gc_vals[np.logical_and(gc_vals[:,0] == Maxg_nc[0], gc_vals[:,1] == Maxg_nc[1])]
            #print(np.unique(nterm_data[:,0]),np.unique(nterm_data[:,1]),np.unique(nterm_data[:,2]),np.unique(nterm_data[:,3]))
            #print(np.unique(cterm_data[:,0]),np.unique(cterm_data[:,1]),np.unique(cterm_data[:,2]),np.unique(cterm_data[:,3]))
            #print(f'Size of gn vals in loop_ij data for nc {Maxg_nc}: {nterm_data.shape}')
            #print(f'Size of gc vals in loop_ij data for nc {Maxg_nc}: {cterm_data.shape}')

            if loop_dom == thread_dom:
                nterm_data = nterm_data[np.where(nterm_data[:,3] < Maxg_nc[0])]
                cterm_data = cterm_data[np.where(cterm_data[:,3] > Maxg_nc[1])]
                #print(f'Size of gn vals in loop_ij data for nc {Maxg_nc}: {nterm_data.shape}')
                #print(f'Size of gc vals in loop_ij data for nc {Maxg_nc}: {cterm_data.shape}')
                #print(nterm_data[:10],nterm_data[-10:])
                #print(cterm_data)
            else:
                pass

            if nterm_data.shape[0] != 0:
                #print('nterm')
                if loop_dom == thread_dom:
                    win_start_res = Maxg_nc[0] - 4
                elif loop_dom != thread_dom:
                    win_start_res = total_dom_cooridx_dict[thread_dom][0]

                win_end_res = win_start_res - window_size #non inclusive
                win_middle_res = win_start_res - int(window_size/2)
                first_nterm_res = nterm_data[0,-2]
                #print(nterm_data[-10:], nterm_data.shape)
                outdata = []

                #catch if window is to large for tail length
                #in this case just return the maxg in the tail
                #with the middle residue of the tail
                if win_end_res < first_nterm_res:
                    #print('window to big for tail')
                    max_g = max(np.abs(nterm_data[:,-1]))
                    win_middle_res =int(((win_start_res - first_nterm_res)/2) + first_nterm_res)
                    outdata.append([win_middle_res,max_g])
                    #print(win_start_res, win_middle_res, win_end_res, first_nterm_res, max_g)

                while win_end_res >= first_nterm_res:

                    window_data = nterm_data[np.where(np.logical_and(nterm_data[:,-2] <= win_start_res, nterm_data[:,-2] > win_end_res))]
                    #print(window_data)
                    if window_data.shape[0] == 0:
                        max_g = 0
                    else:
                        if ent_res_method == 1:
                            max_g = max(np.abs(window_data[:,-1]))
                        elif ent_res_method == 2 or ent_res_method == 3:
                            max_g = np.abs(np.sum(window_data[:,-1]))/(4*np.pi)
                    outdata.append([win_middle_res,max_g])
                    #print(win_start_res, win_middle_res, win_end_res, first_nterm_res, max_g)

                    win_start_res -= 1
                    win_end_res -= 1 #non inclusive
                    win_middle_res -= 1

                #print(f'outdata: {outdata}')
                if len(outdata) != 0:
                    nterm_outdata = np.stack(outdata)
                    #print(f'nterm_outdata: {nterm_outdata}')


            if cterm_data.shape[0] != 0:
                #print('cterm')
                #print(cterm_data)
                if loop_dom == thread_dom:
                    win_start_res = Maxg_nc[1] + 5
                elif loop_dom != thread_dom:
                    win_start_res = total_dom_cooridx_dict[thread_dom][0]

                win_end_res = win_start_res + window_size #non inclusive
                win_middle_res = win_start_res + int(window_size/2)
                last_cterm_res = cterm_data[-1,-2]

                #print(cterm_data[-10:])
                outdata = []

                #catch if window is to large for tail length
                #in this case just return the maxg in the tail
                #with the middle residue of the tail
                if win_end_res > last_cterm_res:
                    #print('window to big for tail')
                    max_g = max(np.abs(cterm_data[:,-1]))
                    win_middle_res =int(((last_cterm_res - win_start_res)/2) + last_cterm_res)
                    outdata.append([win_middle_res,max_g])

                while win_end_res <= last_cterm_res:
                    window_data = cterm_data[np.where(np.logical_and(cterm_data[:,-2] >= win_start_res, cterm_data[:,-2] < win_end_res))]
                    #print(window_data.shape)
                    if window_data.shape[0] == 0:
                        max_g = 0
                    else:
                        if ent_res_method == 1:
                            max_g = max(np.abs(window_data[:,-1]))
                        elif ent_res_method == 2 or ent_res_method == 3:
                            max_g = np.abs(np.sum(window_data[:,-1]))/(4*np.pi)
                    outdata.append([win_middle_res,max_g])
                    #print(win_start_res, win_middle_res, win_end_res, last_cterm_res, max_g)
                    #print(window_data[np.where(np.abs(window_data[:,-1]) == max_g)])

                    win_start_res += 1
                    win_end_res += 1 #non inclusive
                    win_middle_res += 1

                if len(outdata) != 0:
                    cterm_outdata = np.stack(outdata)
                    #print(f'cterm_outdata: {cterm_outdata}')

            #combine data for native contact
            if (nterm_data.shape[0] != 0 and cterm_data.shape[0] != 0):
                #print(f'both tails analyzed')
                total_term_data = np.vstack((nterm_outdata, cterm_outdata))
                #print(f'total_term_data.shape: {total_term_data.shape}')
            elif nterm_data.shape[0] != 0:
                #print(f'ntail only analyzed')
                total_term_data = nterm_outdata
                #print(f'total_term_data.shape: {total_term_data.shape}')
            elif cterm_data.shape[0] != 0:
                #print(f'ctail only analyzed')
                total_term_data = cterm_outdata
                #print(f'total_term_data.shape: {total_term_data.shape}')

            try:
                del nterm_outdata
            except:
                pass
            try:
                del cterm_outdata
            except:
                pass

            #print(f'total_term_data: {total_term_data}')
            #plt.hist(total_term_data[:,1])
            #plt.savefig(f'test_hist.png')


            gmax = np.max(total_term_data[:,1])
            #print(f'gmax for terminal residues near entanglement: {gmax}')

            if ent_res_method == 1:
                if gmax >= 0.5:
                    gmax_res = total_term_data[np.where(total_term_data[:,1] == gmax)]
                    gmax_res = gmax_res[:,0].astype(int)

                    gmax_res_list.append([Maxg_nc,gmax_res])
                    gmax_res_set = gmax_res_set.union(gmax_res)

            elif ent_res_method == 2:
                gmax_res = total_term_data[np.where(total_term_data[:,1] == gmax)]
                gmax_res = gmax_res[:,0].astype(int)

                gmax_res_list.append([Maxg_nc,gmax_res])
                gmax_res_set = gmax_res_set.union(gmax_res)

            elif ent_res_method == 3:

                total_term_data_95th_percentile = np.percentile(total_term_data[:,1],95)
                #total_term_data_95th_percentile_data = total_term_data[np.where(total_term_data[:,1] >= total_term_data_95th_percentile)]
                #print(f'total_term_data_95th_percentile: {total_term_data_95th_percentile}')
                #print(f'total_term_data_95th_percentile_data: {total_term_data_95th_percentile_data}')

                gmax_res = total_term_data[np.where(total_term_data[:,1] >= total_term_data_95th_percentile)]
                gmax_res = gmax_res[:,0].astype(int)

                gmax_res_list.append([Maxg_nc,gmax_res])
                gmax_res_set = gmax_res_set.union(gmax_res)

        #print(f'gmax_res_list: {gmax_res_list}')

        if loop_dom == thread_dom:
            #print(loop_dom,thread_dom,corid2resid_dom_dict,gmax_res_set)
            gmax_res_set = set(corid2resid_dom_dict[loop_dom][res] for res in gmax_res_set)
        elif loop_dom != thread_dom:
            gmax_res_set = set(corid2resid_dom_dict['total'][res] for res in gmax_res_set)

        gmax_res_set_dom_pair_dict[dom_pair] = gmax_res_set #in resid
        gmax_res_list_dom_pair_dict[dom_pair] = gmax_res_list #in corid


    Gr_dom_pair_out_dict = {}
    for dom_pair,nc_Gr in gmax_res_list_dom_pair_dict.items():
        #print(f'Gr{"".join(dom_pair)}_ent_corid: {nc_Gr}')
        thread_dom = dom_pair[0]
        loop_dom = dom_pair[1]
        if loop_dom != thread_dom:
            loop_dom = 'total'

        Gr_temp = {}
        for k,v in nc_Gr:
            rnc = (corid2resid_dom_dict[loop_dom][k[0]],corid2resid_dom_dict[loop_dom][k[1]])
            rnc_rset = [corid2resid_dom_dict[loop_dom][x] for x in v]
            Gr_temp[rnc] = rnc_rset
            #print(rnc,rnc_rset)
        Gr_dom_pair_out_dict[dom_pair] = Gr_temp

    #print('\n',Gr_dom_pair_out_dict)
    return  gmax_res_set_dom_pair_dict, Gr_dom_pair_out_dict
    #return  gmax_res_set_dom_pair_dict, gmax_res_list_dom_pair_dict

def stability_score(dom_pair_Maxg_dict, Maxg_nc_dict, bt_pot_cmap_dict, Gr_list_dict):
    if print_framesummary: print(f'\nCalculating stability score')
    S_dict = {}

    for dom_pair,Maxg_dict in Maxg_nc_dict.items():
        thread_dom = dom_pair[0]
        loop_dom = dom_pair[1]
        if dom_pair_Maxg_dict[dom_pair] == 0:
            S_dict[dom_pair] = [0, 0, 0]
            continue

        if dom_pair_Maxg_dict[dom_pair] == None:
            S_dict[dom_pair] = [0, 0, 0]
            continue

        #estimate average loop closing contact potential
        contacts = np.asarray(list(Maxg_dict.keys()))

        if loop_dom == thread_dom:
            contact_pot = bt_pot_cmap_dict[dom_pair[1]][contacts]
        else:
            contact_pot = bt_pot_cmap_dict['total'][contacts]

        S1 = (-1)*np.sum(contact_pot)/len(contact_pot)

        #estimate average thread flux
        S2 = []
        Gr_list = Gr_list_dict[dom_pair]

        for Gr_nc,Gr in Gr_list.items():
            Gr_resids = [x.number for x in Gr]
            Gr_median = np.median(Gr_resids)
            if thread_dom == loop_dom:
                if Gr_median < Gr_nc[0].number:
                    Lnterm = Gr_nc[0].number
                    delta_l = Lnterm - Gr_median
                    l = float(delta_l)/float(Lnterm)
                    #print(delta_l, Lnterm, l)
                    S2.append(l)

                if Gr_median > Gr_nc[1].number:
                    max_res = len(dom_ref_structures[thread_dom].residues)
                    #print(f'max_res: {max_res}')
                    Lcterm = max_res - Gr_nc[1].number
                    delta_l = Gr_median - Gr_nc[1].number
                    l = float(delta_l)/float(Lcterm)
                    #print(delta_l, Lcterm, l)
                    S2.append(l)
            else:
                size_loop_dom = bt_pot_cmap_dict[loop_dom].shape[0]
                delta_l = min(Gr_median-size_loop_dom, size_loop_dom-Gr_median)
                l = float(1)/float(delta_l)
                S2.append(l)

        S2 = np.asarray(S2)
        S2 = np.sum(S2)/len(S2)

        S = S1/S2

        S_dict[dom_pair] = [S, S1, S2]

    return S_dict


def struc_overlap(ref_cmap_dict, frame_cmap_dict):


    dom_struc_overlap_dict = {}

    for dom in frame_cmap_dict.keys():
        ref_dmap = ref_cmap_dict[dom][1]
        ref_ndists = ref_cmap_dict[dom][2]
        frame_dmap = frame_cmap_dict[dom][1]

        diff_matrix = np.full_like(frame_dmap,0.2*3.81)
        diff_matrix  = diff_matrix - (frame_dmap - ref_dmap)

        if resid_masks != None:

            diff_matrix=np.triu(diff_matrix,k=4)

            if len(corid_masks) == 0:
                print(f'ERROR: len(corid_masks) == 0')
                quit()


            dom_corid_mask = corid_masks[dom][-1]
            anti_dom_corid_mask = [x for x in np.arange(0,diff_matrix.shape[0]) if x not in dom_corid_mask]
            diff_matrix[anti_dom_corid_mask,:] = 0
            diff_matrix[:,anti_dom_corid_mask] = 0

            diff_matrix[diff_matrix > 0] = 1
            diff_matrix[diff_matrix < 0] = 0
            dists_formed = diff_matrix.sum()

            dom_struc_overlap_dict[dom] = float(dists_formed)/float(ref_ndists)

        elif resid_masks == None:
            print(f'if struc_overlap = yes a resid_masks needs to be specified')
            sys.exit()

    return dom_struc_overlap_dict


######################################################################################################################
# MAIN                                                                                                               #
######################################################################################################################
print(f'\nInit finished, staring MAIN')
### START make reference structure and various data structures ###

#generate ref state contact, resid, resname, BT contact potential arrays
bt_array, aa_names_to_bt_array_idx = bt_parse(bt_contact_pot_file)

#make iteraiable of all pairs of residue sturcture objs
res_pair_struct_dict = {}
for dom,struct in dom_ref_structures.items():
    print(dom,struct)
    max_res = len(struct.residues)
    res_pair_struct_dict[dom] = np.asarray(list(itertools.product(struct, struct))).reshape((max_res,max_res,2))

#ref_fraction_native_contact_analysis
if fraction_native_contact_analysis:
    ref_cmap_restricted_dict = cmap(dom_ref_structures, ref=True, restricted=True, cut_off = 8.0, bb_buffer = 4)

#ref_struct_overlap_analysis
if struct_overlap_analysis:
    ref_cmap_struct_overlap_restricted_dict = cmap(dom_ref_structures, ref=True, restricted=True, cut_off = 999.0, bb_buffer = 4)

#ref_entanglement_analysis
if entanglement_analysis or fraction_native_contacts_wchngent_analysis:
    ref_cmap_dict = cmap(dom_ref_structures, ref=True, restricted=False)

    ref_bt_pot_cmap_dict = bt_pot_cmap(bt_array, aa_names_to_bt_array_idx, ref_cmap_dict, res_pair_struct_dict)

    #generate ref_nc_gdict
    ref_dom_pair_nc_gdict, ref_dom_pair_gn_vals, ref_dom_pair_gc_vals = gen_nc_gdict(dom_ref_structures, ref_cmap_dict, termini_threshold)

    #generate Maxg dictionary for ref domain pairs
    print('generate Maxg dictionary for ref')
    ref_dom_pair_Maxg_dict = {}
    ref_dom_pair_Maxg_nc_dict = {}
    for dom_pair,ref_nc_gdict in ref_dom_pair_nc_gdict.items():
        print('\n',dom_pair,ref_nc_gdict)
        ref_Maxg = max(abs(np.asarray(list(ref_nc_gdict.values()))))
        print(f'ref_Maxg: {ref_Maxg}')
        ref_dom_pair_Maxg_dict[dom_pair] = ref_Maxg

        ref_Maxg_dict = {k:v for k,v in ref_nc_gdict.items() if abs(v) == ref_Maxg}
        print(f'ref_Maxg_dict: {ref_Maxg_dict}')
        ref_dom_pair_Maxg_nc_dict[dom_pair] = ref_Maxg_dict

    if entanglement_analysis:
        #generate Gr for ref
        ref_Gr_dom_pair_dict, ref_Gr_list_dom_pair_dict = Gr(ref_dom_pair_Maxg_dict, ref_dom_pair_Maxg_nc_dict, ref_dom_pair_gn_vals, ref_dom_pair_gc_vals)
        #print(f'ref_Gr_dom_pair_dict: {ref_Gr_dom_pair_dict}')
        print(f'\nref_Gr_list_dom_pair_dict:\n')
        for k,v in ref_Gr_list_dom_pair_dict.items():
            print(k)
            for nc,grlist in v.items():
                print(nc,grlist)

        #generate S for ref
        ref_S_dict = stability_score(ref_dom_pair_Maxg_dict,ref_dom_pair_Maxg_nc_dict, ref_bt_pot_cmap_dict, ref_Gr_list_dom_pair_dict)
        print(f'ref_S_dict: {ref_S_dict}')

print(f'Finished making reference structure object and maps')
if ref_only == True:
    sys.exit()
### END make reference structure and various data structures ###

##> make a flag for stopping here for single PDB analysis

### START loading of analysis universe ###
print(f'\nStarting trajectory analysis')
#get alpha carbons atoms and then positions of them
u = Universe(psf,in_paths)
if model_res == 'aa':
    u_calphas = u.select_atoms('name CA')

elif model_res == 'cg':
    u_calphas = u.select_atoms('all')

else:
    print(f'ERROR: only aa and cg model_res supported.')
    quit()

#print(u_calphas)
### END loading of analysis universe ###

### START analysis of universe ###
outdata = []
outdata_labels = []
outdata_dtypes = []
print_framesummary
for ts in u.trajectory[start_frame:end_frame:frame_stride]:
    if print_framesummary: print(f'\n\nFrame: {ts.frame}')

    #load frame_coor onto pmd structures
    for dom,struct in dom_ref_structures.items():
        #u_calphas_prm = u_calphas.convert_to('PARMED')
        frame_coor = u_calphas.positions
        if len(total_dom_cooridx_dict) > 0:
            frame_coor = frame_coor[total_dom_cooridx_dict[dom]]
        frame_res = u_calphas.residues

        temp_dict = {}
        for coor,res in zip(frame_coor,frame_res):
            resid = res.resid
            new_coorid = resid - 1
            temp_dict[new_coorid] = [res,coor]

        for coorid,struct_parts in temp_dict.items():
            res = struct_parts[0]
            coor = struct_parts[1]
            struct.atoms[coorid].xx = coor[0]
            struct.atoms[coorid].xy = coor[1]
            struct.atoms[coorid].xz = coor[2]
            struct.residues[coorid] = res
        #print(struct.coordinates[:10])

    #initilize output data structures and prime them with frame and time
    output = [ts.frame, ts.time]
    output_labels = ['frame_num', 'time']
    output_dtypes = ['int','float']
    if print_framesummary: print(f'\nframe_num, time: {output}')

    #fraction_native_contact_analsyis
    if fraction_native_contact_analysis:
        #frame contact and distance map creation and correction for thermal flux
        #frame_cmap, frame_dmap, frame_cnum = cmap(frame_coor, ref=False, ref_distance_map=ref_dmap, ref_contact_map=ref_cmap)
        frame_cmap_restricted_dict = cmap(dom_ref_structures, ref=False, ref_cmap_restricted_dict=ref_cmap_restricted_dict, restricted=True)

        #frame fraction of native contacts Q
        if print_framesummary: print(f'\nQ analsysis')
        for dom in frame_cmap_restricted_dict.keys():
            #print(dom)
            frame_cnum = frame_cmap_restricted_dict[dom][-1]
            ref_cnum = ref_cmap_restricted_dict[dom][-1]
            Q = float(frame_cnum)/float(ref_cnum)
            #print(frame_cnum,ref_cnum,Q)

            if resid_masks !=  None:
                mask_label = corid_masks[dom][0]
                if print_framesummary: print(mask_label)
            else:
                mask_label = ''

            #print(f'Q{dom}{mask_label}: {Q}')
            output += [Q]
            output_labels += [f'Q{dom}{mask_label}']
            output_dtypes += ['float']

    #frame_struct_overlap_analysis
    if struct_overlap_analysis:
        frame_cmap_struct_overlap_restricted_dict = cmap(dom_ref_structures, ref=True, restricted=True, cut_off = 999.0, bb_buffer = 4)

        chi_dict = struc_overlap(ref_cmap_struct_overlap_restricted_dict,frame_cmap_struct_overlap_restricted_dict)
        #print(f'chi_dict: {chi_dict}')

        if len(chi_dict) != 0:
            for dom,chi in chi_dict.items():

                if resid_masks !=  None:
                    mask_label = corid_masks[dom][0]
                    if print_framesummary: print(mask_label)
                else:
                    mask_label = ''

                output += [chi]
                output_labels += [f'chi{dom}{mask_label}']
                output_dtypes += ['float']

    #RMSD analysis
    if rmsd_analysis:
        if print_framesummary: print(f'RMSD analysis')
        for dom,ref_coor in dom_ref_structures_coor.items():
            if print_framesummary: print(dom)
            if rmsd_masks_dom_dict != None and rmsd_masks_dom_coorid_dict != None:
                #print(rmsd_masks_dom_coorid_dict, len(rmsd_masks_dom_coorid_dict[dom]))
                frame_coor = dom_ref_structures[dom].coordinates
                frame_coor = frame_coor[rmsd_masks_dom_coorid_dict[dom]]
                #print(frame_coor[:10],frame_coor.shape)
                ref_coor = ref_coor[rmsd_masks_dom_coorid_dict[dom]]
                #print(ref_coor[:10],ref_coor.shape)
            else:
                frame_coor = dom_ref_structures[dom].coordinates
            rmsd_val = MDAnalysis.analysis.rms.rmsd(ref_coor, frame_coor, superposition = True)

            if print_framesummary: print(f'RMSD{dom}: {rmsd_val}')
            output += [rmsd_val]
            output_labels += [f'RMSD{dom}']
            output_dtypes += ['float']


    #entanglement analysis
    if entanglement_analysis or fraction_native_contacts_wchngent_analysis:
        frame_cmap_dict = cmap(dom_ref_structures, ref=False, ref_cmap_dict=ref_cmap_dict, restricted=False)
        #generate frame_nc_gdict
        #frame_nc_gdict, frame_gn_vals, frame_gc_vals = gen_nc_gdict(frame_coor, max_res, termini_threshold, frame_cmap)
        #generate frame_nc_gdict
        frame_dom_pair_nc_gdict, frame_dom_pair_gn_vals, frame_dom_pair_gc_vals = gen_nc_gdict(dom_ref_structures, frame_cmap_dict, termini_threshold)
        #print(f'frame_dom_pair_nc_gdict: {frame_dom_pair_nc_gdict}')

        #find changes in entanglement
        frame_dom_pair_chng_ent_dict = gen_chng_ent_dict(ref_dom_pair_nc_gdict, frame_dom_pair_nc_gdict)

        if print_framesummary: print(f'frame_dom_pair_chng_ent_dict: {frame_dom_pair_chng_ent_dict}')

        #frame fraction change in entanglement
        for dom_pair in frame_dom_pair_chng_ent_dict.keys():
            ref_cnum = 0
            joined_dom_pair = ''.join(dom_pair)

            if dom_pair[0] != dom_pair[1]:
                for dom in dom_pair:
                    ref_cnum += ref_cmap_dict[dom][-1]
                #print(f'ref_cnum for G{joined_dom_pair}: {ref_cnum}')
            elif dom_pair[0] == dom_pair[1]:
                ref_cnum += ref_cmap_dict[dom_pair[0]][-1]
                #print(f'ref_cnum for G{joined_dom_pair}: {ref_cnum}')


            num_nc_wchngent = len(frame_dom_pair_chng_ent_dict[dom_pair].keys())
            #print(f'num_nc_wchngent: {num_nc_wchngent}')

            G = float(num_nc_wchngent)/float(ref_cnum)

            if print_framesummary: print(f'G{joined_dom_pair}: {G}')

            output += [G]
            output_labels += [f'G{joined_dom_pair}']
            output_dtypes += ['float']


        if entanglement_analysis:
            #generate Maxg for frame
            frame_dom_pair_Maxg_dict = {}
            frame_dom_pair_Maxg_nc_dict = {}
            frame_dom_pair_chng_ent_Maxg_dict = {}
            frame_dom_pair_chng_ent_Maxg_nc_dict = {}
            for dom_pair,frame_nc_gdict in frame_dom_pair_nc_gdict.items():

                #all native contacts
                try:
                    frame_Maxg = max(abs(np.asarray(list(frame_nc_gdict.values()))))
                except:
                    frame_Maxg = 0

                if print_framesummary: print(f'\nframe_Maxg{dom_pair}: {frame_Maxg}')

                output += [frame_Maxg]
                output_labels += [f'frame_Maxg{"".join(dom_pair)}']
                output_dtypes += ['int']
                frame_dom_pair_Maxg_dict[dom_pair] = frame_Maxg

                #get frame_nc_gdict with Maxg
                frame_Maxg_dict = {k:v for k,v in frame_nc_gdict.items() if abs(v) == frame_Maxg}
                ##print(f'frame_Maxg_dict{dom_pair}: {frame_Maxg_dict}')
                if dom_pair[0] == dom_pair[1]:
                    frame_Maxg_dict_resid = {(corid2resid_dom_dict[dom_pair[0]][k[0]],corid2resid_dom_dict[dom_pair[0]][k[1]]):v for k,v in frame_Maxg_dict.items()}
                if dom_pair[0] != dom_pair[1]:
                    frame_Maxg_dict_resid = {(corid2resid_dom_dict['total'][k[0]],corid2resid_dom_dict['total'][k[1]]):v for k,v in frame_Maxg_dict.items()}
                output += [frame_Maxg_dict_resid]
                output_labels += [f'frame_Maxg_dict_resid{"".join(dom_pair)}']
                output_dtypes += ['O']

                if frame_Maxg != 0:
                    if print_framesummary: print(f'frame_Maxg_dict_resid{dom_pair}: {frame_Maxg_dict_resid}')

                frame_dom_pair_Maxg_nc_dict[dom_pair] = frame_Maxg_dict

                #only native contacts with change in ent
                if frame_dom_pair_chng_ent_dict[dom_pair].values():
                    frame_Maxg_wchngent = max(abs(np.asarray(list(frame_dom_pair_chng_ent_dict[dom_pair].values()))[:,1]))
                    if print_framesummary: print(f'frame_Maxg_wchngent{dom_pair}: {frame_Maxg_wchngent}')
                    output += [frame_Maxg_wchngent]
                    output_labels += [f'frame_Maxg_wchngent{"".join(dom_pair)}']
                    output_dtypes += ['int']
                    frame_dom_pair_chng_ent_Maxg_dict[dom_pair] = frame_Maxg_wchngent

                    #get frame_chng_gdict with Maxg
                    frame_Maxg_wchngent_dict = {k:v for k,v in frame_dom_pair_chng_ent_dict[dom_pair].items() if abs(v[1]) == frame_Maxg_wchngent}

                    if print_framesummary: print(f'frame_Maxg_wchngent_dict{dom_pair}: {frame_Maxg_wchngent_dict}')

                    frame_dom_pair_chng_ent_Maxg_nc_dict[dom_pair] = frame_Maxg_wchngent_dict
                    if dom_pair[0] == dom_pair[1]:
                        frame_Maxg_wchngent_dict_resid = {(corid2resid_dom_dict[dom_pair[0]][k[0]],corid2resid_dom_dict[dom_pair[0]][k[1]]):v for k,v in frame_Maxg_wchngent_dict.items()}
                    if dom_pair[0] != dom_pair[1]:
                        frame_Maxg_wchngent_dict_resid = {(corid2resid_dom_dict['total'][k[0]],corid2resid_dom_dict['total'][k[1]]):v for k,v in frame_Maxg_wchngent_dict.items()}
                    #print(f'frame_Maxg_wchngent_dict_resid{dom_pair}: {frame_Maxg_wchngent_dict_resid}')
                    output += [frame_Maxg_wchngent_dict_resid]
                    output_labels += [f'frame_Maxg_wchngent_dict_resid{"".join(dom_pair)}']
                    output_dtypes += ['O']

                #no change in ent for dom_pair
                else:
                    frame_Maxg_wchngent = 0
                    output += [frame_Maxg_wchngent]
                    output_labels += [f'frame_Maxg_wchngent{"".join(dom_pair)}']
                    output_dtypes += ['int']
                    frame_dom_pair_chng_ent_Maxg_dict[dom_pair] = frame_Maxg_wchngent

                    frame_dom_pair_chng_ent_Maxg_nc_dict[dom_pair] = {}
                    frame_Maxg_wchngent_dict_resid = set()
                    output += [frame_Maxg_wchngent_dict_resid]
                    output_labels += [f'frame_Maxg_wchngent_dict_resid{"".join(dom_pair)}']
                    output_dtypes += ['O']

            #generate Gr for frame
            if print_framesummary: print(f'\nGr analysis all native contacts')

            frame_Gr_dom_pair_dict, frame_Gr_list_dom_pair_dict = Gr(frame_dom_pair_Maxg_dict, frame_dom_pair_Maxg_nc_dict, frame_dom_pair_gn_vals, frame_dom_pair_gc_vals)

            frame_Gr_dom_pair_out_dict = {}
            for dom_pair,nc_Gr in frame_Gr_list_dom_pair_dict.items():
                if print_framesummary:
                    print(f'frame_Gr{dom_pair}:')
                    for nc,Gr_res in nc_Gr.items():
                        print(nc,Gr_res)

                thread_dom = dom_pair[0]
                loop_dom = dom_pair[1]
                if loop_dom != thread_dom:
                    loop_dom = 'total'


                output += [nc_Gr]
                output_labels += [f'frame_Gr{"".join(dom_pair)}']
                output_dtypes += ['O']


            #print(f'\nGr analysis only native contacts with chng in ent')
            frame_Gr_dom_pair_wchngent_dict, frame_Gr_dom_pair_wchngent_list_dict = Gr(frame_dom_pair_chng_ent_Maxg_dict, frame_dom_pair_chng_ent_Maxg_nc_dict, frame_dom_pair_gn_vals, frame_dom_pair_gc_vals)

            frame_Gr_dom_pair_wchngent_out_dict = {}
            for dom_pair,nc_Gr in frame_Gr_dom_pair_wchngent_list_dict.items():
                if print_framesummary:
                    print(f'frame_Gr_wchngent{dom_pair}:')
                    for nc,Gr_res in nc_Gr.items():
                        print(nc,Gr_res)
                thread_dom = dom_pair[0]
                loop_dom = dom_pair[1]
                if loop_dom != thread_dom:
                    loop_dom = 'total'

                output += [nc_Gr]
                output_labels += [f'frame_Gr_wchngent{"".join(dom_pair)}']
                output_dtypes += ['O']

            #generate S for frame
            frame_S_dict = stability_score(frame_dom_pair_Maxg_dict,frame_dom_pair_Maxg_nc_dict, ref_bt_pot_cmap_dict, frame_Gr_list_dom_pair_dict)
            #print(f'frame_S_dict: {frame_S_dict}')
            for dom,S in frame_S_dict.items():
                output += [S[0], S[1], S[2]]
                output_labels += [f'frame_S{"".join(dom)}', f'frame_S1{"".join(dom)}', f'frame_S2{"".join(dom)}']
                output_dtypes += ['float', 'float', 'float']

            frame_S_wchngent_dict = stability_score(frame_dom_pair_chng_ent_Maxg_dict,frame_dom_pair_chng_ent_Maxg_nc_dict, ref_bt_pot_cmap_dict, frame_Gr_list_dom_pair_dict)
            #print(f'frame_S_wchngent_dict: {frame_S_wchngent_dict}')
            for dom,S in frame_S_wchngent_dict.items():
                output += [S[0], S[1], S[2]]
                output_labels += [f'frame_S_wchngent{"".join(dom)}', f'frame_S1_wchngent{"".join(dom)}', f'frame_S2_wchngent{"".join(dom)}']
                output_dtypes += ['float', 'float', 'float']

            #calculate discrete Gk prob dist for each
            #print(f'\nCalculate discrete Gk prob dist')
            for dom,frame_chng_ent_dict in frame_dom_pair_chng_ent_dict.items():
                #print(dom,frame_chng_ent_dict)
                #generate freqGk
                ks = np.arange(0,5)
                if frame_chng_ent_dict:
                    Gks = np.asarray(list(frame_chng_ent_dict.values()))[:,0]
                    freqGk, _ = np.histogram(Gks, bins=np.arange(-0.5,5.5,1))
                    #output += [freqGk]
                    output += [Gk for Gk in freqGk]
                    #output_labels += [[f'probG{k}' for k in ks]]
                    output_labels += [f'freqG{k}{"".join(dom)}' for k in ks]
                    output_dtypes += ['int' for k in ks]
                else:
                    output += [0,0,0,0,0]
                    #output_labels += [[f'probG{k}' for k in ks]]
                    output_labels += [f'freqG{k}{"".join(dom)}' for k in ks]
                    output_dtypes += ['int' for k in ks]


    ##print(output, output_labels)
    outdata.append(output)

#check if data was generated
if len(outdata) == 0:
    print(f'No data was generated. Check if you specified an analysis or if your DCD was empty. Exitting...')
    sys.exit()

outdata = np.asarray(outdata, dtype='O')
outdata_dict = {}
for idx,label in enumerate(output_labels):
    #print(idx,label)
    data = outdata[:,idx]
    dtype = output_dtypes[idx]
    outdata_dict[label] = pd.Series(data, dtype=dtype)
    #print(outdata_dict)

outdf = pd.DataFrame(outdata_dict)
print(outdf)
### END analysis of universe ###
### END output ###
outdf.to_pickle(f'{out_path}{script_title}_{script_version}/output/{outfile_basename}_s{start_frame}_e{end_frame}_m{frame_stride}.df')
print(f'Saved: {out_path}{script_title}_{script_version}/output/{outfile_basename}_s{start_frame}_e{end_frame}_m{frame_stride}.df')
#pd.set_option('display.max_columns', 100)
#pd.set_option('display.max_rows', 100)
######################################################################################################################
comp_time=time.time()-start_time
print(f'computation time: {comp_time}')
