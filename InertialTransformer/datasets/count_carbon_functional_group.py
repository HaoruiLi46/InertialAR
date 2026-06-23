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

def count_carboaromatics(mol):
    """
    Count pure benzene, naphthalene, and anthracene/phenanthrene ring systems.

    Returns:
        (count_benzene, count_naphthalene, count_anthracene_phenanthrene)
    """
    if not mol:
        return 0, 0, 0

    p_benzene = Chem.MolFromSmarts('c1ccccc1')
    p_naphthalene = Chem.MolFromSmarts('c1ccc2ccccc2c1')
    p_anthracene = Chem.MolFromSmarts('c1ccc2cc3ccccc3cc2c1')
    p_phenanthrene = Chem.MolFromSmarts('c1ccc2c(c1)ccc3ccccc32')

    raw_benzene = len(mol.GetSubstructMatches(p_benzene))
    raw_naphthalene = len(mol.GetSubstructMatches(p_naphthalene))
    raw_anthracene = len(mol.GetSubstructMatches(p_anthracene))
    raw_phenanthrene = len(mol.GetSubstructMatches(p_phenanthrene))

    count_three_ring = raw_anthracene + raw_phenanthrene
    count_naphthalene = raw_naphthalene - (2 * count_three_ring)
    count_benzene = raw_benzene - (2 * count_naphthalene) - (3 * count_three_ring)

    count_benzene = max(0, count_benzene)
    count_naphthalene = max(0, count_naphthalene)
    
    return count_benzene, count_naphthalene, count_three_ring

def count_alkene_cc(mol):
    """
    Count isolated alkenes and conjugated alkene systems with mutually exclusive SMARTS.
    
    Returns:
        (isolated_alkene, pure_diene, pure_triene, pure_tetraene)
    """
    if not mol: return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    p_all_alkene = Chem.MolFromSmarts('[C!a]=[C!a]')
    count_all_alkene = len(mol.GetSubstructMatches(p_all_alkene))
    p_isolated_alkene = Chem.MolFromSmarts('[C!a;!$(C-C=C);!$(C-a)]=[C!a;!$(C-C=C);!$(C-a)]')
    p_pure_diene = Chem.MolFromSmarts('[C!a;!$(C-C=C);!$(C-a)]=[C!a]-[C!a]=[C!a;!$(C-C=C);!$(C-a)]')
    p_pure_triene = Chem.MolFromSmarts('[C!a;!$(C-C=C);!$(C-a)]=[C!a]-[C!a]=[C!a]-[C!a]=[C!a;!$(C-C=C);!$(C-a)]')
    p_pure_tetraene = Chem.MolFromSmarts('[C!a;!$(C-C=C);!$(C-a)]=[C!a]-[C!a]=[C!a]-[C!a]=[C!a]-[C!a]=[C!a;!$(C-C=C);!$(C-a)]')

    p_cyclohexene = Chem.MolFromSmarts('C1=CCCCC1')
    p_cyclopentene = Chem.MolFromSmarts('C1=CCCC1')
    p_cycloheptene = Chem.MolFromSmarts('C1=CCCCCC1')
    p_cyclopentadiene = Chem.MolFromSmarts('C1=CC=CC1')
    p_13_Cyclohexadiene = Chem.MolFromSmarts('C1=CC=CCC1')
    p_14_Cyclohexadiene = Chem.MolFromSmarts('C1=CCC=CC1')
    count_cyclohexene = len(mol.GetSubstructMatches(p_cyclohexene)) 
    count_cyclopentene = len(mol.GetSubstructMatches(p_cyclopentene)) 
    count_cycloheptene = len(mol.GetSubstructMatches(p_cycloheptene)) 
    count_cyclopentadiene = len(mol.GetSubstructMatches(p_cyclopentadiene)) 
    count_13_Cyclohexadiene = len(mol.GetSubstructMatches(p_13_Cyclohexadiene)) 
    count_14_Cyclohexadiene = len(mol.GetSubstructMatches(p_14_Cyclohexadiene)) 
    count_isolated_alkene = len(mol.GetSubstructMatches(p_isolated_alkene)) - count_cyclohexene - count_cyclopentene - count_cycloheptene - count_14_Cyclohexadiene
    count_pure_diene = len(mol.GetSubstructMatches(p_pure_diene)) - count_cyclopentadiene - count_13_Cyclohexadiene
    count_pure_triene = len(mol.GetSubstructMatches(p_pure_triene))
    count_pure_tetraene = len(mol.GetSubstructMatches(p_pure_tetraene))
    p_alkene_with_aromatic = Chem.MolFromSmarts('[a]-[C!a]=[C!a;!$(C-C=C);!$(C-a)]')
    p_diene_with_aromatic = Chem.MolFromSmarts('[a]-[C!a]=[C!a]-[C!a]=[C!a;!$(C-C=C);!$(C-a)]')
    p_triene_with_aromatic = Chem.MolFromSmarts('[a]-[C!a]=[C!a]-[C!a]=[C!a]-[C!a]=[C!a;!$(C-C=C);!$(C-a)]')
    p_tetraene_with_aromatic = Chem.MolFromSmarts('[a]-[C!a]=[C!a]-[C!a]=[C!a]-[C!a]=[C!a]-[C!a]=[C!a;!$(C-C=C);!$(C-a)]')
    p_alkene_with_di_aromatic = Chem.MolFromSmarts('[a]-[C!a]=[C!a]-[a]')
    p_diene_with_di_aromatic = Chem.MolFromSmarts('[a]-[C!a]=[C!a]-[C!a]=[C!a]-[a]')

    count_alkene_with_aromatic = len(mol.GetSubstructMatches(p_alkene_with_aromatic))
    count_diene_with_aromatic = len(mol.GetSubstructMatches(p_diene_with_aromatic))
    count_triene_with_aromatic = len(mol.GetSubstructMatches(p_triene_with_aromatic))
    count_tetraene_with_aromatic = len(mol.GetSubstructMatches(p_tetraene_with_aromatic))
    count_alkene_with_di_aromatic = len(mol.GetSubstructMatches(p_alkene_with_di_aromatic))
    count_diene_with_di_aromatic = len(mol.GetSubstructMatches(p_diene_with_di_aromatic))
    if (count_isolated_alkene + count_pure_diene + count_pure_triene + count_pure_tetraene 
        + count_alkene_with_aromatic + count_diene_with_aromatic + count_triene_with_aromatic + count_tetraene_with_aromatic 
        + count_alkene_with_di_aromatic + count_diene_with_di_aromatic
        + count_cyclohexene + count_cyclopentene + count_cycloheptene + count_cyclopentadiene + count_13_Cyclohexadiene + count_14_Cyclohexadiene) == 0:
        count_other_alkene = count_all_alkene
    else:
        count_other_alkene = 0

    return count_isolated_alkene, count_pure_diene, count_pure_triene, count_pure_tetraene, count_alkene_with_aromatic, count_diene_with_aromatic, count_triene_with_aromatic, count_tetraene_with_aromatic, count_alkene_with_di_aromatic, count_diene_with_di_aromatic, count_cyclohexene, count_cyclopentene, count_cycloheptene, count_cyclopentadiene, count_13_Cyclohexadiene, count_14_Cyclohexadiene, count_other_alkene

def count_alkyne_cc(mol):
    """Count alkyne C#C bonds with SMARTS C#C."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('C#C')
    return len(mol.GetSubstructMatches(pattern))


CARBON_GROUP_HEADERS = [
    'isolated_alkene',
    'pure_diene', #not in top500
    'pure_triene', #not in top500
    'pure_tetraene', #not in top500
    'alkene_with_aromatic',
    'diene_with_aromatic', #not in top500
    'triene_with_aromatic', #not in top500
    'tetraene_with_aromatic', #not in top500
    'alkene_with_di_aromatic', #not in top500
    'diene_with_di_aromatic', #not in top500
    'cyclohexene', #not in top500
    'cyclopentene',  #not in top500
    'cycloheptene', #not in top500
    'cyclopentadiene',  #not in top500
    'cyclohexadiene_13', #not in top500
    'cyclohexadiene_14', #not in top500
    'other_alkene', #not in top500
    'alkyne_cc',  #not in top500
    'benzene',
    'naphthalene',
    'anthracene_phenanthrene', #not in top500
    
]
def count_carbon_functional_group(mol):
    isolated_alkene, pure_diene, pure_triene, pure_tetraene, alkene_with_aromatic, diene_with_aromatic, triene_with_aromatic, tetraene_with_aromatic, alkene_with_di_aromatic, diene_with_di_aromatic, cyclohexene, cyclopentene, cycloheptene, cyclopentadiene, cyclohexadiene_13, cyclohexadiene_14, other_alkene = count_alkene_cc(mol)
    alkyne_cc = count_alkyne_cc(mol)
    benzene, naphthalene, anthracene_phenanthrene = count_carboaromatics(mol)
    carbon_functional_group_counts = [
        isolated_alkene,
        pure_diene,
        pure_triene,
        pure_tetraene,
        alkene_with_aromatic,
        diene_with_aromatic,
        triene_with_aromatic,
        tetraene_with_aromatic,
        alkene_with_di_aromatic,
        diene_with_di_aromatic,
        cyclohexene,
        cyclopentene,
        cycloheptene,
        cyclopentadiene,
        cyclohexadiene_13,
        cyclohexadiene_14,
        other_alkene,
        alkyne_cc,
        benzene,
        naphthalene,
        anthracene_phenanthrene,
    ]
    return carbon_functional_group_counts


CARBON_GROUP_HEADERS_v2 = [
    'isolated_alkene',
    'conjugated_alkene', # pure_diene, pure_triene, pure_tetraene
    'alkene_with_aromatic',
    'cyclic_alkene',
    'other_alkene', #not in top500
    'alkyne_cc',  #not in top500
    'benzene',
    'naphthalene_anthracene_phenanthrene', #anthracene_phenanthrene not in top500
]
def count_carbon_functional_group_v2(mol):
    isolated_alkene, pure_diene, pure_triene, pure_tetraene, alkene_with_aromatic, diene_with_aromatic, triene_with_aromatic, tetraene_with_aromatic, alkene_with_di_aromatic, diene_with_di_aromatic, cyclohexene, cyclopentene, cycloheptene, cyclopentadiene, cyclohexadiene_13, cyclohexadiene_14, other_alkene = count_alkene_cc(mol)
    alkyne_cc = count_alkyne_cc(mol)
    benzene, naphthalene, anthracene_phenanthrene = count_carboaromatics(mol)
    naphthalene_anthracene_phenanthrene = naphthalene+anthracene_phenanthrene
    carbon_functional_group_counts = [
        isolated_alkene,
        pure_diene+pure_triene+pure_tetraene,
        alkene_with_aromatic+diene_with_aromatic+triene_with_aromatic+tetraene_with_aromatic+alkene_with_di_aromatic+diene_with_di_aromatic,
        cyclohexene+cyclopentene+cycloheptene+cyclopentadiene+cyclohexadiene_13+cyclohexadiene_14,
        other_alkene,
        alkyne_cc,
        benzene,
        naphthalene_anthracene_phenanthrene,
    ]
    return carbon_functional_group_counts


CARBON_GROUP_HEADERS_v3 = [
    'isolated_alkene',
    'conjugated_alkene', # pure_diene, pure_triene, pure_tetraene
    'alkene_with_aromatic',
    'cyclic_alkene',
    'other_alkene', #not in top500
    'alkyne_cc',  #not in top500
    'benzene',
    'naphthalene_anthracene_phenanthrene', #anthracene_phenanthrene not in top500
]
def count_carbon_functional_group_v3(mol):
    isolated_alkene, pure_diene, pure_triene, pure_tetraene, alkene_with_aromatic, diene_with_aromatic, triene_with_aromatic, tetraene_with_aromatic, alkene_with_di_aromatic, diene_with_di_aromatic, cyclohexene, cyclopentene, cycloheptene, cyclopentadiene, cyclohexadiene_13, cyclohexadiene_14, other_alkene = count_alkene_cc(mol)
    alkyne_cc = count_alkyne_cc(mol)
    benzene, naphthalene, anthracene_phenanthrene = count_carboaromatics(mol)
    naphthalene_anthracene_phenanthrene = naphthalene+anthracene_phenanthrene
    carbon_functional_group_counts = [
        isolated_alkene,
        pure_diene+pure_triene+pure_tetraene,
        alkene_with_aromatic+diene_with_aromatic+triene_with_aromatic+tetraene_with_aromatic+alkene_with_di_aromatic+diene_with_di_aromatic,
        cyclohexene+cyclopentene+cycloheptene+cyclopentadiene+cyclohexadiene_13+cyclohexadiene_14,
        other_alkene,
        alkyne_cc,
        benzene,
        naphthalene_anthracene_phenanthrene,
    ]
    return carbon_functional_group_counts


CARBON_GROUP_HEADERS_v5 = [
    'isolated_alkene',
    'conjugated_alkene', # pure_diene, pure_triene, pure_tetraene
    'alkene_with_aromatic',
    'cyclic_alkene',
    'other_alkene', #not in top500
    'alkyne_cc',  #not in top500
    'benzene',
    'naphthalene',
    'anthracene_phenanthrene', #anthracene_phenanthrene not in top500
]
def count_carbon_functional_group_v5(mol):
    isolated_alkene, pure_diene, pure_triene, pure_tetraene, alkene_with_aromatic, diene_with_aromatic, triene_with_aromatic, tetraene_with_aromatic, alkene_with_di_aromatic, diene_with_di_aromatic, cyclohexene, cyclopentene, cycloheptene, cyclopentadiene, cyclohexadiene_13, cyclohexadiene_14, other_alkene = count_alkene_cc(mol)
    alkyne_cc = count_alkyne_cc(mol)
    benzene, naphthalene, anthracene_phenanthrene = count_carboaromatics(mol)
    carbon_functional_group_counts = [
        isolated_alkene,
        pure_diene+pure_triene+pure_tetraene,
        alkene_with_aromatic+diene_with_aromatic+triene_with_aromatic+tetraene_with_aromatic+alkene_with_di_aromatic+diene_with_di_aromatic,
        cyclohexene+cyclopentene+cycloheptene+cyclopentadiene+cyclohexadiene_13+cyclohexadiene_14,
        other_alkene,
        alkyne_cc,
        benzene,
        naphthalene,
        anthracene_phenanthrene,
    ]
    return carbon_functional_group_counts


CARBON_GROUP_HEADERS_B3LYP = [
    'isolated_alkene',
    'pure_diene', #not in top500
    'pure_triene', #not in top500
    'pure_tetraene', #not in top500
    'alkene_with_aromatic',
    'diene_with_aromatic', #not in top500
    'triene_with_aromatic', #not in top500
    'tetraene_with_aromatic', #not in top500
    'alkene_with_di_aromatic', #not in top500
    'diene_with_di_aromatic', #not in top500
    'cyclohexene', #not in top500
    'cyclopentene',  #not in top500
    'cycloheptene', #not in top500
    'cyclopentadiene',  #not in top500
    'cyclohexadiene_13', #not in top500
    'cyclohexadiene_14', #not in top500
    'other_alkene', #not in top500
    'alkyne_cc',  #not in top500
    'benzene',
    'naphthalene',
    'anthracene_phenanthrene', #not in top500
    
]
def count_carbon_functional_group_B3LYP(mol):
    isolated_alkene, pure_diene, pure_triene, pure_tetraene, alkene_with_aromatic, diene_with_aromatic, triene_with_aromatic, tetraene_with_aromatic, alkene_with_di_aromatic, diene_with_di_aromatic, cyclohexene, cyclopentene, cycloheptene, cyclopentadiene, cyclohexadiene_13, cyclohexadiene_14, other_alkene = count_alkene_cc(mol)
    alkyne_cc = count_alkyne_cc(mol)
    benzene, naphthalene, anthracene_phenanthrene = count_carboaromatics(mol)
    carbon_functional_group_counts = [
        isolated_alkene,
        pure_diene,
        pure_triene,
        pure_tetraene,
        alkene_with_aromatic,
        diene_with_aromatic,
        triene_with_aromatic,
        tetraene_with_aromatic,
        alkene_with_di_aromatic,
        diene_with_di_aromatic,
        cyclohexene,
        cyclopentene,
        cycloheptene,
        cyclopentadiene,
        cyclohexadiene_13,
        cyclohexadiene_14,
        other_alkene,
        alkyne_cc,
        benzene,
        naphthalene,
        anthracene_phenanthrene,
    ]
    return carbon_functional_group_counts
