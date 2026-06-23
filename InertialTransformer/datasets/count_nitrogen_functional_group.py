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

def count_secondary_amine(mol): # classify_by_ring_membership
    """
    Classify secondary amines as acyclic, endocyclic, exocyclic, or inter-ring.
    Args:
        mol: RDKit molecule.
    Returns:
        tuple: (acyclic_count, endocyclic_count, exocyclic_count, inter_ring_count)
    """
    if not mol: return 0, 0, 0, 0

    pattern = Chem.MolFromSmarts('[NH1]([#6;!$(C=O)])[#6;!$(C=O)]')
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return 0, 0, 0, 0

    counts = defaultdict(int)

    for match in matches:
        nitrogen_idx = match[0]
        
        if mol.GetAtomWithIdx(nitrogen_idx).IsInRing():
            counts['Endocyclic_Amine'] += 1
            continue

        is_acyclic = all(not mol.GetAtomWithIdx(idx).IsInRing() for idx in match)
        if is_acyclic:
            counts['Acyclic_Amine'] += 1
        else:
            counts['Inter-ring_Amine'] += 1

    pattern_special = Chem.MolFromSmarts('[NH1]([#6;!$(C=O);!R])[#6;!$(C=O);R]')
    matches_special = mol.GetSubstructMatches(pattern_special)
    if matches_special:
        counts['Exocyclic_Amine'] += len(matches_special)
        counts['Inter-ring_Amine'] -= len(matches_special)

    return counts['Acyclic_Amine'], counts['Endocyclic_Amine'], counts['Exocyclic_Amine'], counts['Inter-ring_Amine']

def count_tertiary_amine(mol): # classify_by_ring_membership
    """
    Classify tertiary amines as acyclic, endocyclic, exocyclic, or inter-ring.
    Args:
        mol: RDKit molecule.
    Returns:
        tuple: (acyclic_count, endocyclic_count, exocyclic_count, inter_ring_count)
    """
    if not mol: return 0, 0, 0, 0

    pattern = Chem.MolFromSmarts('[N;H0;X3;+0]([#6;!$(C=O)])([#6;!$(C=O)])[#6;!$(C=O)]')
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return 0, 0, 0, 0 

    counts = defaultdict(int)

    for match in matches:
        nitrogen_idx = match[0]
        
        if mol.GetAtomWithIdx(nitrogen_idx).IsInRing():
            counts['Endocyclic_Amine'] += 1
            continue

        is_acyclic = all(not mol.GetAtomWithIdx(idx).IsInRing() for idx in match)
        if is_acyclic:
            counts['Acyclic_Amine'] += 1
        else:
            counts['Inter-ring_Amine'] += 1

    pattern_special = Chem.MolFromSmarts('[N;H0;X3;+0]([#6;!$(C=O);!R])([#6;!$(C=O);!R])[#6;!$(C=O);R]')
    matches_special = mol.GetSubstructMatches(pattern_special)
    if matches_special:
        counts['Exocyclic_Amine'] += len(matches_special)
        counts['Inter-ring_Amine'] -= len(matches_special)

    return counts['Acyclic_Amine'], counts['Endocyclic_Amine'], counts['Exocyclic_Amine'], counts['Inter-ring_Amine']

def count_primary_amine(mol):
    """
    Count primary amines with SMARTS [NH2][#6;!$(C=O)].
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[NH2][#6;!$(C=O)]')
    return len(mol.GetSubstructMatches(pattern))

def count_amide(mol):
    """Count amides with SMARTS [#6](=O)[N;!$(N-N);!$(N-O)]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6](=O)[N;!$(N-N);!$(N-O)]')
    return len(mol.GetSubstructMatches(pattern))

def count_urea(mol):
    """Count ureas with SMARTS [N;!$(N-N);!$(N-O)][#6](=O)[N;!$(N-N);!$(N-O)]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[N;!$(N-N);!$(N-O)][#6](=O)[N;!$(N-N);!$(N-O)]')
    return len(mol.GetSubstructMatches(pattern))

def count_nitrile(mol):
    """Count nitriles with SMARTS C#N."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('C#N')
    return len(mol.GetSubstructMatches(pattern))

def count_nitro(mol):
    """Count nitro groups with SMARTS [N+](=O)[O-]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[N+](=O)[O-]')
    return len(mol.GetSubstructMatches(pattern))

def count_hydrazine(mol):
    """Count hydrazine-like N-N single bonds with SMARTS [#6][NX3][NX3][#6]."""
    if not mol:return 0
    pattern = Chem.MolFromSmarts('[#6][NX3][NX3][#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_acylhydrazone(mol):
    """Count acylhydrazones with SMARTS [#6!a](=[O])[N;X3][N;X2]=[C]."""
    if not mol:return 0
    pattern = Chem.MolFromSmarts('[#6!a](=[O])[N;X3][N;X2]=[C]')
    return len(mol.GetSubstructMatches(pattern))

def count_n_phenyl_imine(mol):
    """Count N-phenyl imines with SMARTS [a]~[N;X2]=[C;X3]."""
    if not mol:return 0
    pattern = Chem.MolFromSmarts('[a]~[N;X2]=[C;X3]')
    return len(mol.GetSubstructMatches(pattern))

NITROGEN_GROUP_HEADERS = [
    'secondary_acyclic_exocyclic_amine',
    'tertiary_acyclic_exocyclic_amine',
    'endocyclic_amine',
    'inter_ring_amine', # not in top 500
    'primary_amine',
    'amide',
    'urea',
    'nitrile', # not in top 500
    'nitro',
    'hydrazine', # not in top 500
    'acylhydrazone',
    'n_phenyl_imine',
]

def count_nitrogen_functional_group(mol):
    secondary_acyclic, secondary_endocyclic, secondary_exocyclic, secondary_inter_ring = count_secondary_amine(mol)
    tertiary_acyclic, tertiary_endocyclic, tertiary_exocyclic, tertiary_inter_ring = count_tertiary_amine(mol)
    urea = count_urea(mol)
    nitrogen_functional_group_counts = [ secondary_acyclic+secondary_exocyclic,
        tertiary_acyclic+tertiary_exocyclic,
        secondary_endocyclic+tertiary_endocyclic,
        secondary_inter_ring+tertiary_inter_ring,
        count_primary_amine(mol),
        count_amide(mol)-2*urea,
        urea,
        count_nitrile(mol),
        count_nitro(mol),
        count_hydrazine(mol),
        count_acylhydrazone(mol),
        count_n_phenyl_imine(mol),
    ]
    nitrogen_atom_count = sum(nitrogen_functional_group_counts)
    nitrogen_atom_count += nitrogen_functional_group_counts[NITROGEN_GROUP_HEADERS.index('hydrazine')]
    nitrogen_atom_count += nitrogen_functional_group_counts[NITROGEN_GROUP_HEADERS.index('acylhydrazone')]
    nitrogen_atom_count += nitrogen_functional_group_counts[NITROGEN_GROUP_HEADERS.index('urea')]
    return nitrogen_functional_group_counts, nitrogen_atom_count


NITROGEN_GROUP_HEADERS_v2 = [
    'secondary_acyclic_exocyclic_inter_ring_amine',
    'tertiary_acyclic_exocyclic_inter_ring_amine',
    'endocyclic_amine',
    'primary_amine',
    'amide',
    'urea',
    'nitro',
    'acylhydrazone',
    'n_phenyl_imine',
    'nitrile', # not in top 500
    'hydrazine', # not in top 500
]

def count_nitrogen_functional_group_v2(mol):
    secondary_acyclic, secondary_endocyclic, secondary_exocyclic, secondary_inter_ring = count_secondary_amine(mol)
    tertiary_acyclic, tertiary_endocyclic, tertiary_exocyclic, tertiary_inter_ring = count_tertiary_amine(mol)
    primary_amine = count_primary_amine(mol)
    amide = count_amide(mol)
    urea = count_urea(mol)
    nitro = count_nitro(mol)
    acylhydrazone = count_acylhydrazone(mol)
    n_phenyl_imine = count_n_phenyl_imine(mol)
    nitrile = count_nitrile(mol)
    hydrazine = count_hydrazine(mol)
    nitrogen_functional_group_counts = [ 
        secondary_acyclic+secondary_exocyclic+secondary_inter_ring,
        tertiary_acyclic+tertiary_exocyclic+tertiary_inter_ring,
        secondary_endocyclic+tertiary_endocyclic,
        primary_amine,
        amide-2*urea,
        urea,
        nitro,
        acylhydrazone,
        n_phenyl_imine,
        nitrile,
        hydrazine,
    ]
    nitrogen_atom_count = sum(nitrogen_functional_group_counts)
    nitrogen_atom_count += hydrazine
    nitrogen_atom_count += acylhydrazone
    nitrogen_atom_count += urea
    
    oxygen_atom_count = 0
    oxygen_atom_count += amide # amide has one O;
    oxygen_atom_count += urea # urea has one O;
    oxygen_atom_count += 2*nitro # nitro has two O;
    oxygen_atom_count += acylhydrazone # acylhydrazone has one O;
    return nitrogen_functional_group_counts, nitrogen_atom_count, oxygen_atom_count


NITROGEN_GROUP_HEADERS_v3 = [
    'amine',
    'amide',
    'urea',
    'nitro',
    'acylhydrazone',
    'n_phenyl_imine',
    'nitrile', # not in top 500
]

def count_nitrogen_functional_group_v3(mol):
    secondary_acyclic, secondary_endocyclic, secondary_exocyclic, secondary_inter_ring = count_secondary_amine(mol)
    tertiary_acyclic, tertiary_endocyclic, tertiary_exocyclic, tertiary_inter_ring = count_tertiary_amine(mol)
    primary_amine = count_primary_amine(mol)
    amide = count_amide(mol)
    urea = count_urea(mol)
    nitro = count_nitro(mol)
    acylhydrazone = count_acylhydrazone(mol)
    n_phenyl_imine = count_n_phenyl_imine(mol)
    nitrile = count_nitrile(mol)
    amine = (secondary_acyclic+secondary_endocyclic+secondary_exocyclic+secondary_inter_ring
            +tertiary_acyclic+tertiary_endocyclic+tertiary_exocyclic+tertiary_inter_ring
            +primary_amine)
    nitrogen_functional_group_counts = [ 
        amine,
        amide-2*urea,
        urea,
        nitro,
        acylhydrazone,
        n_phenyl_imine,
        nitrile,
    ]
    nitrogen_atom_count = sum(nitrogen_functional_group_counts)
    nitrogen_atom_count += acylhydrazone
    nitrogen_atom_count += urea
    
    oxygen_atom_count = 0
    oxygen_atom_count += amide # amide has one O;
    oxygen_atom_count += urea # urea has one O;
    oxygen_atom_count += 2*nitro # nitro has two O;
    oxygen_atom_count += acylhydrazone # acylhydrazone has one O;
    return nitrogen_functional_group_counts, nitrogen_atom_count, oxygen_atom_count


NITROGEN_GROUP_HEADERS_B3LYP = [
    'secondary_acyclic_exocyclic_amine',
    'tertiary_acyclic_exocyclic_amine',
    'endocyclic_amine',
    'inter_ring_amine',
    'primary_amine',
    'amide',
    'urea',
    'nitro',
    'acylhydrazone',
    'n_phenyl_imine',
    'hydrazine',
    'nitrile', 
]

def count_nitrogen_functional_group_B3LYP(mol):
    secondary_acyclic, secondary_endocyclic, secondary_exocyclic, secondary_inter_ring = count_secondary_amine(mol)
    tertiary_acyclic, tertiary_endocyclic, tertiary_exocyclic, tertiary_inter_ring = count_tertiary_amine(mol)
    primary_amine = count_primary_amine(mol)
    amide = count_amide(mol)
    urea = count_urea(mol)
    nitro = count_nitro(mol)
    acylhydrazone = count_acylhydrazone(mol)
    n_phenyl_imine = count_n_phenyl_imine(mol)
    nitrile = count_nitrile(mol)
    hydrazine = count_hydrazine(mol)
    
    secondary_acyclic_exocyclic_amine=secondary_acyclic+secondary_exocyclic
    tertiary_acyclic_exocyclic_amine=tertiary_acyclic+tertiary_exocyclic
    endocyclic_amine=secondary_endocyclic+tertiary_endocyclic
    inter_ring_amine=secondary_inter_ring+tertiary_inter_ring

    nitrogen_functional_group_counts = [ 
        secondary_acyclic_exocyclic_amine,
        tertiary_acyclic_exocyclic_amine,
        endocyclic_amine,
        inter_ring_amine,
        primary_amine,
        amide-2*urea,
        urea,
        nitro,
        acylhydrazone,
        n_phenyl_imine,
        nitrile,
        hydrazine,
    ]
    nitrogen_atom_count = sum(nitrogen_functional_group_counts)
    nitrogen_atom_count += hydrazine
    nitrogen_atom_count += acylhydrazone
    nitrogen_atom_count += urea
    
    oxygen_atom_count = 0
    oxygen_atom_count += amide # amide has one O;
    oxygen_atom_count += urea # urea has one O;
    oxygen_atom_count += 2*nitro # nitro has two O;
    oxygen_atom_count += acylhydrazone # acylhydrazone has one O;
    return nitrogen_functional_group_counts, nitrogen_atom_count, oxygen_atom_count
