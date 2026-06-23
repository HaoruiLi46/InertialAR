import csv
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from collections import Counter
import json
import torch
from collections import defaultdict
def count_thioether(mol):
    """
    Count thioethers with SMARTS [#6;!$(C=O)][SD2][#6;!$(C=O)].
    Carbonyl carbons are excluded to avoid thioesters.
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6;!$(C=O)][SD2][#6;!$(C=O)]')
    return len(mol.GetSubstructMatches(pattern))

def count_thiol(mol):
    """
    Count thiols with SMARTS [#6;!$(C=O)][SH1].
    Carbonyl carbons are excluded to avoid thio-carboxylic acids.
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6;!$(C=O)][SH1]')
    return len(mol.GetSubstructMatches(pattern))

def count_disulfide(mol):
    """Count disulfides with SMARTS [#6][SD2][SD2][#6]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6][SD2][SD2][#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_sulfonamide(mol):
    """Count sulfonamides with SMARTS [#6]S(=O)(=O)[N;!$(N-N)]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6]S(=O)(=O)[N;!$(N-N)]')
    return len(mol.GetSubstructMatches(pattern))

def count_sulfonylhydrazone(mol):
    """Count sulfonylhydrazones with SMARTS [#6]S(=O)(=O)[N;X3][N;X2]=[C]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6]S(=O)(=O)[N;X3][N;X2]=[C]')
    return len(mol.GetSubstructMatches(pattern))

def count_sulfonic_acid(mol):
    """Count sulfonic acids with SMARTS [#6]S(=O)(=O)[OH]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6]S(=O)(=O)[OH]')
    return len(mol.GetSubstructMatches(pattern))

def count_sulfoxide(mol):
    """Count sulfoxides with SMARTS [#6][S;X3](=O)[#6]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6][S;X3](=O)[#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_sulfone(mol):
    """Count sulfones with SMARTS [#6][S;X4](=O)(=O)[#6]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6][S;X4](=O)(=O)[#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_thioester(mol):
    """Count thioesters with SMARTS [#6]C(=O)S[#6]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6]C(=O)S[#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_thioaldehyde_or_thioketone(mol):
    """
    Count thioaldehydes and thioketones by combining C(=S)H and C(=S)C patterns.
    """
    if not mol: return 0
    patt_general_thial = Chem.MolFromSmarts('[CH1](=S)[#6]')
    patt_thioformaldehyde = Chem.MolFromSmarts('[CH2]=S')
    
    patt_thioketone = Chem.MolFromSmarts('[#6]C(=S)[#6]')
    
    count = len(mol.GetSubstructMatches(patt_general_thial)) + \
            len(mol.GetSubstructMatches(patt_thioformaldehyde)) + \
            len(mol.GetSubstructMatches(patt_thioketone))
    return count

def count_sulfonate(mol):
    """Count sulfonates with SMARTS [#6]S(=O)(=O)O[#6]."""
    if not mol: return 0
    
    pattern = Chem.MolFromSmarts('[#6]S(=O)(=O)O[#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_thioamide(mol):
    """Count thioamides with SMARTS [#6](=S)[N;!$(N-N);!$(N-O)]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6](=S)[N;!$(N-N);!$(N-O)]')
    return len(mol.GetSubstructMatches(pattern))

def count_thiourea(mol):
    """Count thioureas with SMARTS [N;!$(N-N);!$(N-O)][#6](=S)[N;!$(N-N);!$(N-O)]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[N;!$(N-N);!$(N-O)][#6](=S)[N;!$(N-N);!$(N-O)]')
    return len(mol.GetSubstructMatches(pattern))

SULFUR_GROUP_HEADERS = [
    'thioether',
    'thiol', # not in top 500
    'disulfide', # not in top 500
    'sulfonamide',
    'sulfonylhydrazone',
    'sulfonic_acid', # not in top 500
    'sulfoxide',
    'sulfone', # not in top 500
    'thioester',  # not in top 500
    'thioaldehyde_or_thioketone',  # not in top 500
    'sulfonate', # not in top 500
    'thioamide', 
    'thiourea',
]

def count_sulfur_functional_group(mol):
    thioether = count_thioether(mol)
    thiol = count_thiol(mol)
    disulfide = count_disulfide(mol)
    sulfonamide = count_sulfonamide(mol)
    sulfonylhydrazone = count_sulfonylhydrazone(mol)
    sulfonic_acid = count_sulfonic_acid(mol)
    sulfoxide = count_sulfoxide(mol)
    sulfone = count_sulfone(mol)
    thioester = count_thioester(mol)
    thioaldehyde_or_thioketone = count_thioaldehyde_or_thioketone(mol)
    sulfonate = count_sulfonate(mol)
    thiourea = count_thiourea(mol)
    thioamide = count_thioamide(mol) - 2*thiourea
    
    sulfur_functional_group_counts = [
        thioether,
        thiol,
        disulfide,
        sulfonamide,
        sulfonylhydrazone,
        sulfonic_acid,
        sulfoxide,
        sulfone,
        thioester,
        thioaldehyde_or_thioketone,
        sulfonate,
        thioamide,
        thiourea,
    ]
    sulfur_atom_count = sum(sulfur_functional_group_counts)
    sulfur_atom_count += sulfur_functional_group_counts[SULFUR_GROUP_HEADERS.index('disulfide')]
    return sulfur_functional_group_counts, sulfur_atom_count


SULFUR_GROUP_HEADERS_v2 = [
    'thioether_thiol', # 'thiol' not in top 500
    'disulfide', # not in top 500
    'sulfonamide',
    'sulfonylhydrazone',
    'sulfoxide_sulfone',
    'thioester',  # not in top 500
    'thioaldehyde_or_thioketone',  # not in top 500
    'sulfonate_sulfonic_acid', # not in top 500
    'thioamide', 
    'thiourea',
]

def count_sulfur_functional_group_v2(mol):
    thioether = count_thioether(mol)
    thiol = count_thiol(mol)
    disulfide = count_disulfide(mol)
    sulfonamide = count_sulfonamide(mol)
    sulfonylhydrazone = count_sulfonylhydrazone(mol)
    sulfonic_acid = count_sulfonic_acid(mol)
    sulfoxide = count_sulfoxide(mol)
    sulfone = count_sulfone(mol)
    thioester = count_thioester(mol)
    thioaldehyde_or_thioketone = count_thioaldehyde_or_thioketone(mol)
    sulfonate = count_sulfonate(mol)
    thiourea = count_thiourea(mol)
    thioamide = count_thioamide(mol) - 2*thiourea
    
    sulfur_functional_group_counts = [
        thioether+thiol,
        disulfide,
        sulfonamide,
        sulfonylhydrazone,
        sulfoxide+sulfone,
        thioester,
        thioaldehyde_or_thioketone,
        sulfonate+sulfonic_acid,
        thioamide,
        thiourea,
    ]
    sulfur_atom_count = sum(sulfur_functional_group_counts)
    sulfur_atom_count += disulfide # disulfide has two S;
    
    oxygen_atom_count, nitrogen_atom_count = 0, 0
    
    oxygen_atom_count += 2*sulfonamide #sulfonamide has two O;
    nitrogen_atom_count += sulfonamide #sulfonamide has one N;
    
    oxygen_atom_count += 2*sulfonylhydrazone #sulfonylhydrazone has two O;
    nitrogen_atom_count += 2*sulfonylhydrazone #sulfonylhydrazone has two N;
    
    oxygen_atom_count += 3*sulfonic_acid #sulfonic_acid has three O;
    oxygen_atom_count += 3*sulfonate #sulfonate has three O;
    
    oxygen_atom_count += 2*sulfone #sulfone has two O;
    oxygen_atom_count += sulfoxide #sulfoxide has one O;
    oxygen_atom_count += thioester #thioester has one O;
    
    nitrogen_atom_count += thioamide #thioamide has one N;
    nitrogen_atom_count += 2*thiourea #thiourea has two N;

    return sulfur_functional_group_counts, sulfur_atom_count, oxygen_atom_count, nitrogen_atom_count


SULFUR_GROUP_HEADERS_v3 = [
    'thioether_thiol_disulfide', # 'thiol disulfide' not in top 500
    'sulfonamide',
    'sulfonylhydrazone',
    'sulfoxide_sulfone',
    'thioester',  # not in top 500
    'thioaldehyde_or_thioketone',  # not in top 500
    'sulfonate_sulfonic_acid', # not in top 500
    'thioamide', 
    'thiourea',
]

def count_sulfur_functional_group_v3(mol):
    thioether = count_thioether(mol)
    thiol = count_thiol(mol)
    disulfide = count_disulfide(mol)
    sulfonamide = count_sulfonamide(mol)
    sulfonylhydrazone = count_sulfonylhydrazone(mol)
    sulfonic_acid = count_sulfonic_acid(mol)
    sulfoxide = count_sulfoxide(mol)
    sulfone = count_sulfone(mol)
    thioester = count_thioester(mol)
    thioaldehyde_or_thioketone = count_thioaldehyde_or_thioketone(mol)
    sulfonate = count_sulfonate(mol)
    thiourea = count_thiourea(mol)
    thioamide = count_thioamide(mol) - 2*thiourea
    
    sulfur_functional_group_counts = [
        thioether+thiol+disulfide,
        sulfonamide,
        sulfonylhydrazone,
        sulfoxide+sulfone,
        thioester,
        thioaldehyde_or_thioketone,
        sulfonate+sulfonic_acid,
        thioamide,
        thiourea,
    ]
    sulfur_atom_count = sum(sulfur_functional_group_counts)
    sulfur_atom_count += disulfide # disulfide has two S;
    
    oxygen_atom_count, nitrogen_atom_count = 0, 0
    
    oxygen_atom_count += 2*sulfonamide #sulfonamide has two O;
    nitrogen_atom_count += sulfonamide #sulfonamide has one N;
    
    oxygen_atom_count += 2*sulfonylhydrazone #sulfonylhydrazone has two O;
    nitrogen_atom_count += 2*sulfonylhydrazone #sulfonylhydrazone has two N;
    
    oxygen_atom_count += 3*sulfonic_acid #sulfonic_acid has three O;
    oxygen_atom_count += 3*sulfonate #sulfonate has three O;
    
    oxygen_atom_count += 2*sulfone #sulfone has two O;
    oxygen_atom_count += sulfoxide #sulfoxide has one O;
    oxygen_atom_count += thioester #thioester has one O;
    
    nitrogen_atom_count += thioamide #thioamide has one N;
    nitrogen_atom_count += 2*thiourea #thiourea has two N;

    return sulfur_functional_group_counts, sulfur_atom_count, oxygen_atom_count, nitrogen_atom_count
