#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm, trange
from collections import Counter
import json
import torch
from collections import defaultdict
import random
from InertialTransformer.datasets.count_carbon_functional_group import count_carbon_functional_group_v5, CARBON_GROUP_HEADERS_v5
from InertialTransformer.datasets.count_oxygen_functional_group import count_oxygen_functional_group_v3, OXYGEN_GROUP_HEADERS_v3
from InertialTransformer.datasets.count_nitrogen_functional_group import count_nitrogen_functional_group_v3, NITROGEN_GROUP_HEADERS_v3
from InertialTransformer.datasets.count_sulfur_functional_group import count_sulfur_functional_group_v3, SULFUR_GROUP_HEADERS_v3
from InertialTransformer.datasets.count_aliphatic_ring import count_aliphatic_rings_v8, ALIPHATIC_RINGS_v8
from InertialTransformer.datasets.count_aromatic_heterocycles import count_aromatic_heterocycles_extended_v8, AROMATIC_HETEROCYCLES_EXTENDED_v8
from InertialTransformer.datasets.count_halogen import count_halogen_v3, HALOGEN_GROUP_HEADERS_v3


def count_carbon_atoms(mol):
    
    if not mol: return 0
    
    carbon_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            carbon_count += 1
    return carbon_count

def count_aromatic_rings(mol):

    if not mol: return 0

    aromatic_rings = 0
    for ring in Chem.GetSSSR(mol):
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_rings += 1
    return aromatic_rings

def count_nitrogen_atoms(mol):

    if not mol: return 0
    
    nitrogen_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:  
            nitrogen_count += 1
    return nitrogen_count

def count_oxygen_atoms(mol):

    if not mol: return 0
    
    oxygen_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8:  
            oxygen_count += 1
    return oxygen_count

def count_phosphorus_atoms(mol):

    if not mol: return 0
    
    phosphorus_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 15:  
            phosphorus_count += 1
    return phosphorus_count

def count_sulfur_atoms(mol):

    if not mol: return 0
    
    sulfur_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 16:  
            sulfur_count += 1
    return sulfur_count

def count_fluoro_atoms(mol):
    
    if not mol: return 0
    
    fluoro_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 9:
            fluoro_count += 1
    return fluoro_count

def count_chloro_atoms(mol):
    
    if not mol: return 0
    
    chloro_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 17:
            chloro_count += 1
    return chloro_count

def count_bromo_atoms(mol):
    
    if not mol: return 0
    
    bromo_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 35:
            bromo_count += 1
    return bromo_count

def count_iodo_atoms(mol):
    
    if not mol: return 0
    
    iodo_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 53:
            iodo_count += 1
    return iodo_count

def analyze_smiles_file(input_pt_file, output_npy_file, output_header_csv_file):
    """
    Extract the configured functional-group count vector for each SMILES string.
    """
    total_counts = []
    
    # load from smiles.pt file
    print(f"Loading SMILES list from: {input_pt_file}...")
    smiles_dict = torch.load(input_pt_file)
    smiles_list = smiles_dict['smiles']
    print(f"Loaded {len(smiles_list)} SMILES strings.")
    
    max_total_count = 0
    max_carbon_atom_count = 0
    
    name_list = []
    name_list.append('Carbon_Atoms_Count')
    name_list.extend(CARBON_GROUP_HEADERS_v5)
    name_list.extend(ALIPHATIC_RINGS_v8)
    name_list.extend(AROMATIC_HETEROCYCLES_EXTENDED_v8)
    name_list.extend(NITROGEN_GROUP_HEADERS_v3)
    name_list.extend(OXYGEN_GROUP_HEADERS_v3)
    name_list.extend(SULFUR_GROUP_HEADERS_v3)
    name_list.extend(HALOGEN_GROUP_HEADERS_v3)
    name_list.append('Phosphorus_Atoms_Count')
    name_list.append('Rare_Aromatic_FG')
    name_list.append('Rare_Nitrogen_FG')
    name_list.append('Rare_Oxygen_FG')
    name_list.append('Rare_Sulfur_FG')
    name_list.append('Rare_Halogen_FG')
    
    benzene_index = CARBON_GROUP_HEADERS_v5.index('benzene')
    naphthalene_index = CARBON_GROUP_HEADERS_v5.index('naphthalene')
    anthracene_phenanthrene_index = CARBON_GROUP_HEADERS_v5.index('anthracene_phenanthrene')
    for idx, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Analyzing SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            counts = []
            
            total_count = 0
            carbon_atoms_count = count_carbon_atoms(mol)
            aromatic_rings_count = count_aromatic_rings(mol)
            nitrogen_atoms_count = count_nitrogen_atoms(mol)
            oxygen_atoms_count = count_oxygen_atoms(mol)
            phosphorus_atoms_count = count_phosphorus_atoms(mol)
            sulfur_atoms_count = count_sulfur_atoms(mol)
            fluoro_atoms_count = count_fluoro_atoms(mol)
            chloro_atoms_count = count_chloro_atoms(mol)
            bromo_atoms_count = count_bromo_atoms(mol)
            iodo_atoms_count = count_iodo_atoms(mol)
            
            aliphatic_rings_count = count_aliphatic_rings_v8(mol)
            (aromatic_heterocycles_count,
            aromatic_heterocycles_sum,
            n_num_from_aromantic_heterocycle, 
            o_num_from_aromantic_heterocycle, 
            s_num_from_aromantic_heterocycle) = count_aromatic_heterocycles_extended_v8(mol)
            
            carbon_functional_group_count = count_carbon_functional_group_v5(mol)
            oxygen_functional_group_count, o_num_from_O_FG = count_oxygen_functional_group_v3(mol)
            nitrogen_functional_group_count, n_num_from_N_FG, o_num_from_N_FG = count_nitrogen_functional_group_v3(mol)
            sulfur_functional_group_count, s_num_from_S_FG, o_num_from_S_FG, n_num_from_S_FG = count_sulfur_functional_group_v3(mol)
            halogen_functional_group_count, f_num_from_halogen, cl_num_from_halogen, br_num_from_halogen, i_num_from_halogen = count_halogen_v3(mol)
            
            
            counts.append(carbon_atoms_count)
            counts.extend(carbon_functional_group_count)

            counts.extend(aliphatic_rings_count)
            counts.extend(aromatic_heterocycles_count)

            counts.extend(nitrogen_functional_group_count)
            counts.extend(oxygen_functional_group_count)
            counts.extend(sulfur_functional_group_count)
            counts.extend(halogen_functional_group_count)
            counts.append(phosphorus_atoms_count)
            
            if phosphorus_atoms_count > 0:
                counts[1:-1] = [0] * (len(counts) - 2)

            if all(x < 1 for x in counts[1:]):
                counts.append(aromatic_rings_count
                    -aromatic_heterocycles_sum-carbon_functional_group_count[benzene_index]
                    -carbon_functional_group_count[naphthalene_index]
                    -carbon_functional_group_count[anthracene_phenanthrene_index])
            else:
                counts.append(0)
            
            if all(x < 1 for x in counts[1:]):
                counts.append(nitrogen_atoms_count
                    -n_num_from_N_FG-n_num_from_S_FG-n_num_from_aromantic_heterocycle) 
            else:
                counts.append(0)

            if all(x < 1 for x in counts[1:]):
                counts.append(oxygen_atoms_count
                    -o_num_from_O_FG-o_num_from_N_FG-o_num_from_S_FG-o_num_from_aromantic_heterocycle)
            else:
                counts.append(0)
            
            if all(x < 1 for x in counts[1:]):
                counts.append(sulfur_atoms_count
                    -s_num_from_S_FG-s_num_from_aromantic_heterocycle)
            else:
                counts.append(0)
            
            if all(x < 1 for x in counts[1:]):
                counts.append(fluoro_atoms_count-f_num_from_halogen
                            +chloro_atoms_count-cl_num_from_halogen
                            +bromo_atoms_count-br_num_from_halogen
                            +iodo_atoms_count-i_num_from_halogen)
            else:
                counts.append(0)

            total_count = sum(counts)
            
            if total_count > max_total_count:
                max_total_count = total_count
            if carbon_atoms_count > max_carbon_atom_count:
                max_carbon_atom_count = carbon_atoms_count

            total_counts.append(counts)
        else:
            total_counts.append([0] * len(name_list))
            print(f"Warning: failed to parse SMILES({idx}): {smiles}")
    
    np.save(output_npy_file, np.array(total_counts, dtype=np.int32))
    print(f"\nSaved NumPy count array to: {output_npy_file}")
    
    output_csv_file = output_npy_file.replace('.npy', '.csv')
    with open(output_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(name_list)
        for counts in total_counts:
            writer.writerow(counts)
    
    with open(output_header_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(name_list)
    
    print(f"Maximum total functional-group count: {max_total_count}")
    print(f"Maximum carbon atom count: {max_carbon_atom_count}")
    return name_list, total_counts
