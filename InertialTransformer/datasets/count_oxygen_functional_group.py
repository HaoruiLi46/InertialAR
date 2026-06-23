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

def count_ether(mol): # classify_by_ring_membership
    """
    Classify ether matches as acyclic, endocyclic, exocyclic, or inter-ring.
    """
    if not mol: return 0, 0, 0, 0, 0

    pattern = Chem.MolFromSmarts('[C;!$(C=O)][OD2][C;!$(C=O)]')
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return 0, 0, 0, 0, 0
    
    all_ether_count = len(matches)
    atom_rings_sets = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    
    counts = defaultdict(int)

    for match in matches:
        match_set = set(match)
        is_endocyclic = False
        
        for ring_set in atom_rings_sets:
            if match_set.issubset(ring_set):
                is_endocyclic = True
                break
        
        if is_endocyclic:
            counts['Endocyclic_Ether'] += 1
            continue

        is_acyclic = all(not mol.GetAtomWithIdx(idx).IsInRing() for idx in match)
        if is_acyclic:
            counts['Acyclic_Ether'] += 1
        else:
            counts['Inter-ring_Ether'] += 1

    pattern_special = Chem.MolFromSmarts('[C;!$(C=O);R][OD2][C;!$(C=O);!R]')
    matches_special = mol.GetSubstructMatches(pattern_special)
    if matches_special:
        counts['Exocyclic_Ether'] += len(matches_special)
        counts['Inter-ring_Ether'] -= len(matches_special)

    return counts['Acyclic_Ether'], counts['Endocyclic_Ether'], counts['Exocyclic_Ether'], counts['Inter-ring_Ether'], all_ether_count

def count_aryl_ether(mol):
    """
    Count aryl ethers by combining aryl-alkyl and diaryl ether patterns.
    Diaryl matches are divided by two to correct symmetric double matches.
    """
    if not mol: return 0

    patt_aryl_alkyl = Chem.MolFromSmarts('[c]O[C;!$(C=O)]')
    count_aryl_alkyl = len(mol.GetSubstructMatches(patt_aryl_alkyl))

    patt_diaryl = Chem.MolFromSmarts('[c]O[c]')
    count_diaryl = len(mol.GetSubstructMatches(patt_diaryl)) // 2
    
    return count_aryl_alkyl + count_diaryl

def count_alcohol_hydroxyl(mol):
    """Count alcohol hydroxyl groups with SMARTS [OX2H][CX4]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[OX2H][CX4]')
    return len(mol.GetSubstructMatches(pattern))

def count_phenol_hydroxyl(mol):
    """
    Calculates the number of phenol hydroxyl groups in a molecule.
    - SMARTS pattern: [OX2H][c]
    - [OX2H]: Represents a hydroxyl group (-OH).
    - [c]: Represents any aromatic carbon atom.
    - Advantage: This pattern specifically identifies hydroxyl groups directly attached
                 to an aromatic ring, distinguishing them from alcohol hydroxyls.
    """
    if not mol:
        return 0
    pattern = Chem.MolFromSmarts('[OX2H][c]')
    return len(mol.GetSubstructMatches(pattern))



def count_aldehyde(mol):
    """
    Count aldehydes by combining non-formaldehyde R-CHO and formaldehyde patterns.
    """
    if not mol: return 0
    
    patt_general_aldehyde = Chem.MolFromSmarts('[CH1](=O)[#6]')
    patt_formaldehyde = Chem.MolFromSmarts('[CH2]=O')
    
    count = len(mol.GetSubstructMatches(patt_general_aldehyde)) + \
            len(mol.GetSubstructMatches(patt_formaldehyde))
            
    return count

def count_ketone(mol):
    """Count ketones with SMARTS [#6]C(=O)[#6]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6]C(=O)[#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_carboxylic_acid(mol):
    """Count carboxylic acids with SMARTS C(=O)[OH1]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('C(=O)[OH1]')
    return len(mol.GetSubstructMatches(pattern))

def count_ester(mol):
    """Count esters with SMARTS [#6]C(=O)O[#6]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6]C(=O)O[#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_peroxide(mol):
    """Count peroxides with SMARTS [#6][OD2][OD2][#6]."""
    if not mol: return 0
    
    pattern = Chem.MolFromSmarts('[#6][OD2][OD2][#6]')
    
    return len(mol.GetSubstructMatches(pattern))


def count_hemiacetal_acetal(mol):
    """
    Count hemiacetals and acetals, including formaldehyde-derived variants.
    """
    if not mol: return 0
    patt_general = Chem.MolFromSmarts('[#6][CX4H1]([OX2H1])[OD2][#6]')
    patt_formaldehyde = Chem.MolFromSmarts('[CX4H2]([OX2H1])[OD2][#6]')
    count_hemiacetal = len(mol.GetSubstructMatches(patt_general)) + len(mol.GetSubstructMatches(patt_formaldehyde))
    patt_acetal = Chem.MolFromSmarts('[#6][CX4H1]([OD2][#6])[OD2][#6]')
    patt_special_acetal = Chem.MolFromSmarts('[CX4H2]([OD2][#6])[OD2][#6]')
    count_acetal = len(mol.GetSubstructMatches(patt_acetal))
    count_special_acetal = len(mol.GetSubstructMatches(patt_special_acetal))
    count = (count_hemiacetal + count_acetal + count_special_acetal)
    count_ether = count+count_acetal+count_special_acetal
    return count, count_ether, count_hemiacetal

def count_hemiketal_ketal(mol):
    """
    Count hemiketals and ketals with a substituted sp3 carbon center.
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6][CX4H0]([#6])([OX2H1])[OD2][#6]')
    patt_ketal = Chem.MolFromSmarts('[#6][CX4H0]([#6])([OD2][#6])[OD2][#6]')
    count_ketal = len(mol.GetSubstructMatches(patt_ketal))
    count_hemiketal = len(mol.GetSubstructMatches(pattern))
    count = count_hemiketal + count_ketal
    count_ether = count+count_ketal,
    return count, count_ether, count_hemiketal

OXYGEN_GROUP_HEADERS = [
    'alcohol_hydroxyl',
    'phenol_hydroxyl',
    'aldehyde', # not in top 500
    'ketone',
    'carboxylic_acid',
    'ester',
    'acyclic_exocyclic_ether',
    'endocyclic_ether',
    'inter_ring_ether', # not in top200
    'aryl_ether',
    'hemiacetal_acetal', # not in top 500
    'hemiketal_ketal', # not in top 500
    'peroxide', # not in top 500
    'all_ether_count'
]

def count_oxygen_functional_group(mol):
    acyclic_ether, endocyclic_ether, exocyclic_ether, inter_ring_ether, all_ether_count = count_ether(mol)
    alcohol_hydroxyl = count_alcohol_hydroxyl(mol)
    hemiacetal_acetal, ether_count_hemiacetal_acetal, count_hemiacetal = count_hemiacetal_acetal(mol)
    hemiketal_ketal, ether_count_hemiketal_ketal, count_hemiketal = count_hemiketal_ketal(mol)
    if hemiacetal_acetal > 0:
        alcohol_hydroxyl -=count_hemiacetal
        all_ether_count -= hemiacetal_acetal
    if hemiketal_ketal > 0:
        alcohol_hydroxyl -= count_hemiketal
        all_ether_count -= hemiketal_ketal
        
    oxygen_functional_group_counts = [
        alcohol_hydroxyl,
        count_phenol_hydroxyl(mol),
        count_aldehyde(mol),
        count_ketone(mol),
        count_carboxylic_acid(mol),
        count_ester(mol),
        acyclic_ether+exocyclic_ether,
        endocyclic_ether,
        inter_ring_ether,
        count_aryl_ether(mol),
        hemiacetal_acetal,
        hemiketal_ketal,
        count_peroxide(mol),
        all_ether_count
    ]
    oxygen_atom_count = sum(oxygen_functional_group_counts) - all_ether_count
    oxygen_atom_count += oxygen_functional_group_counts[OXYGEN_GROUP_HEADERS.index('peroxide')]
    oxygen_atom_count += oxygen_functional_group_counts[OXYGEN_GROUP_HEADERS.index('ester')]
    oxygen_atom_count += oxygen_functional_group_counts[OXYGEN_GROUP_HEADERS.index('carboxylic_acid')]
    return oxygen_functional_group_counts, oxygen_atom_count

OXYGEN_GROUP_HEADERS_v2 = [
    'alcohol_hydroxyl',
    'phenol_hydroxyl',
    'ketone_aldehyde',
    'carboxylic_acid',
    'ester',
    'acyclic_exocyclic_inter_ring_ether',
    'endocyclic_ether',
    'aryl_ether',
    'hemiacetal_acetal_hemiketal_ketal', # not in top 500
    'peroxide', # not in top 500
]

def count_oxygen_functional_group_v2(mol):
    acyclic_ether, endocyclic_ether, exocyclic_ether, inter_ring_ether, _ = count_ether(mol)
    alcohol_hydroxyl = count_alcohol_hydroxyl(mol)
    hemiacetal_acetal, _, count_hemiacetal = count_hemiacetal_acetal(mol)
    hemiketal_ketal, _, count_hemiketal = count_hemiketal_ketal(mol)
    if hemiacetal_acetal+hemiketal_ketal > 0:
        alcohol_hydroxyl -=(count_hemiacetal+count_hemiketal)
        acyclic_ether = 0
        inter_ring_ether = 0
        endocyclic_ether = 0
        exocyclic_ether = 0
    
    peroxide = count_peroxide(mol)
    oxygen_functional_group_counts = [
        alcohol_hydroxyl,
        count_phenol_hydroxyl(mol),
        count_aldehyde(mol)+count_ketone(mol),
        count_carboxylic_acid(mol),
        count_ester(mol),
        acyclic_ether+exocyclic_ether+inter_ring_ether,
        endocyclic_ether,
        count_aryl_ether(mol),
        hemiacetal_acetal+hemiketal_ketal,
        peroxide,
    ]
    oxygen_atom_count = sum(oxygen_functional_group_counts)
    oxygen_atom_count += peroxide
    oxygen_atom_count += hemiacetal_acetal+hemiketal_ketal
    oxygen_atom_count += oxygen_functional_group_counts[OXYGEN_GROUP_HEADERS_v2.index('ester')]
    oxygen_atom_count += oxygen_functional_group_counts[OXYGEN_GROUP_HEADERS_v2.index('carboxylic_acid')]
    return oxygen_functional_group_counts, oxygen_atom_count


OXYGEN_GROUP_HEADERS_v3 = [
    'alcohol_hydroxyl',
    'phenol_hydroxyl',
    'ketone_aldehyde',
    'carboxylic_acid',
    'ester',
    'acyclic_ether', #acyclic_exocyclic_inter_ring_endocyclic
    'aryl_ether',
]

def count_oxygen_functional_group_v3(mol):
    acyclic_ether, endocyclic_ether, exocyclic_ether, inter_ring_ether, _ = count_ether(mol)
    alcohol_hydroxyl = count_alcohol_hydroxyl(mol)
    hemiacetal_acetal, _, count_hemiacetal = count_hemiacetal_acetal(mol)
    hemiketal_ketal, _, count_hemiketal = count_hemiketal_ketal(mol)
    if hemiacetal_acetal+hemiketal_ketal > 0:
        alcohol_hydroxyl -=(count_hemiacetal+count_hemiketal)
        acyclic_ether = 0
        inter_ring_ether = 0
        endocyclic_ether = 0
        exocyclic_ether = 0
    
    peroxide = count_peroxide(mol)
    oxygen_functional_group_counts = [
        alcohol_hydroxyl,
        count_phenol_hydroxyl(mol),
        count_aldehyde(mol)+count_ketone(mol),
        count_carboxylic_acid(mol),
        count_ester(mol),
        acyclic_ether+exocyclic_ether+inter_ring_ether+endocyclic_ether,
        count_aryl_ether(mol),
        # hemiacetal_acetal+hemiketal_ketal,
        # peroxide,
    ]
    oxygen_atom_count = sum(oxygen_functional_group_counts)
    # oxygen_atom_count += peroxide
    # oxygen_atom_count += hemiacetal_acetal+hemiketal_ketal
    oxygen_atom_count += oxygen_functional_group_counts[OXYGEN_GROUP_HEADERS_v3.index('ester')]
    oxygen_atom_count += oxygen_functional_group_counts[OXYGEN_GROUP_HEADERS_v3.index('carboxylic_acid')]
    return oxygen_functional_group_counts, oxygen_atom_count


OXYGEN_GROUP_HEADERS_B3LYP = [
    'alcohol_hydroxyl',
    'phenol_hydroxyl',
    'ketone',
    'aldehyde',
    'carboxylic_acid',
    'ester',
    'acyclic_exocyclic_ether',
    'inter_ring_ether',
    'endocyclic_ether',
    'aryl_ether',
    'hemiacetal_acetal',
    'hemiketal_ketal',
    'peroxide', 
]

def count_oxygen_functional_group_B3LYP(mol):
    acyclic_ether, endocyclic_ether, exocyclic_ether, inter_ring_ether, _ = count_ether(mol)
    alcohol_hydroxyl = count_alcohol_hydroxyl(mol)
    hemiacetal_acetal, ether_count_hemiacetal_acetal, count_hemiacetal = count_hemiacetal_acetal(mol)
    hemiketal_ketal, ether_count_hemiketal_ketal, count_hemiketal = count_hemiketal_ketal(mol)
    if hemiacetal_acetal+hemiketal_ketal > 0:
        alcohol_hydroxyl -=(count_hemiacetal+count_hemiketal)
        acyclic_ether = 0
        inter_ring_ether = 0
        endocyclic_ether = 0
        exocyclic_ether = 0
    
    peroxide = count_peroxide(mol)
    oxygen_functional_group_counts = [
        alcohol_hydroxyl,
        count_phenol_hydroxyl(mol),
        count_ketone(mol),
        count_aldehyde(mol),
        count_carboxylic_acid(mol),
        count_ester(mol),
        acyclic_ether+exocyclic_ether,
        inter_ring_ether,
        endocyclic_ether,
        count_aryl_ether(mol),
        hemiacetal_acetal,
        hemiketal_ketal,
        peroxide,
    ]
    oxygen_atom_count = sum(oxygen_functional_group_counts)
    oxygen_atom_count += peroxide
    oxygen_atom_count += hemiacetal_acetal+hemiketal_ketal
    oxygen_atom_count += oxygen_functional_group_counts[OXYGEN_GROUP_HEADERS_v2.index('ester')]
    oxygen_atom_count += oxygen_functional_group_counts[OXYGEN_GROUP_HEADERS_v2.index('carboxylic_acid')]
    return oxygen_functional_group_counts, oxygen_atom_count
