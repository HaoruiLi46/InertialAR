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


def count_phosphate_ester(mol):
    """
    Count phosphate esters with SMARTS [#6]O[PX4](=O)(O)(O).
    The pattern requires an R-O-P ester bond and avoids P-C phosphonates.
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[#6]O[PX4](=O)(O)(O)')
    return len(mol.GetSubstructMatches(pattern))

def count_phosphate_anhydride(mol):
    """
    Count phosphate anhydrides with SMARTS [PX4](=O)(O)O[PX4](=O)(O).
    The pattern matches P-O-P linkages without P-C bonds.
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[PX4](=O)(O)O[PX4](=O)(O)')
    return len(mol.GetSubstructMatches(pattern))

def count_phosphine(mol):
    """
    Count phosphines with SMARTS [PX3;H0]([#6])([#6])[#6].
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[PX3;H0]([#6])([#6])[#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_phosphine_oxide(mol):
    """
    Count phosphine oxides with SMARTS [PX4](=O)([#6])([#6])[#6].
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[PX4](=O)([#6])([#6])[#6]')
    return len(mol.GetSubstructMatches(pattern))

def count_phosphonate(mol):
    """
    Count phosphonic acids and phosphonate mono/diesters with mutually exclusive patterns.
    """
    if not mol: return 0
    patt_diester = Chem.MolFromSmarts('[#6][PX4](=O)(O[#6])O[#6]')
    patt_monoester = Chem.MolFromSmarts('[#6][PX4](=O)(O[#6])[OH1]')
    patt_acid = Chem.MolFromSmarts('[#6][PX4](=O)([OH1])[OH1]')
    
    count = len(mol.GetSubstructMatches(patt_diester)) + \
            len(mol.GetSubstructMatches(patt_monoester)) + \
            len(mol.GetSubstructMatches(patt_acid))
            
    return count

def count_phosphite_ester(mol):
    """
    Count phosphite esters across triester and common tautomeric mono/diester forms.
    """
    if not mol: return 0
    patt_triester = Chem.MolFromSmarts('[PX3](O[#6])(O[#6])O[#6]')
    patt_diester_taut = Chem.MolFromSmarts('[PH1](=O)(O[#6])(O[#6])')
    patt_monoester_taut = Chem.MolFromSmarts('[PH1](=O)(O[#6])[OH1]')
    
    count = len(mol.GetSubstructMatches(patt_triester)) + \
            len(mol.GetSubstructMatches(patt_diester_taut)) + \
            len(mol.GetSubstructMatches(patt_monoester_taut))
            
    return count

PHOSPHORUS_GROUP_HEADERS = [
    'phosphate_ester',
    'phosphate_anhydride',
    'phosphine',
    'phosphine_oxide',
    'phosphonate',
    'phosphite_ester',
]

def count_phosphorus_functional_group(mol):
    phosphate_ester = count_phosphate_ester(mol)
    phosphate_anhydride = count_phosphate_anhydride(mol)
    phosphine = count_phosphine(mol)
    phosphine_oxide = count_phosphine_oxide(mol)
    phosphonate = count_phosphonate(mol)
    phosphite_ester = count_phosphite_ester(mol)
    phosphorus_functional_group_counts = [
        phosphate_ester,
        phosphate_anhydride,
        phosphine,
        phosphine_oxide,
        phosphonate,
        phosphite_ester,
    ]
    phosphorus_atom_count = sum(phosphorus_functional_group_counts)
    phosphorus_atom_count += phosphate_anhydride
    return phosphorus_functional_group_counts, phosphorus_atom_count
