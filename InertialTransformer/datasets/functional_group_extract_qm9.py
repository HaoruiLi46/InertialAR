#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from collections import Counter
import json
import torch
import sys
from contextlib import contextmanager

@contextmanager
def redirect_stdout_to_file(filepath=None):
    """
    A context manager to redirect stdout to a file while also printing to the original stdout.
    """
    if not filepath:
        # If no filepath is provided, do nothing.
        yield
        return

    original_stdout = sys.stdout
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Create a Tee object that writes to both the original stdout and the file
            class Tee:
                def write(self, text):
                    original_stdout.write(text)
                    f.write(text)

                def flush(self):
                    original_stdout.flush()
                    f.flush()

            sys.stdout = Tee()
            yield
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

def count_carbon_atoms(mol):
    """Count all carbon atoms in the molecule."""
    if not mol: return 0
    
    carbon_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            carbon_count += 1
    return carbon_count

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

def count_benzene_rings(mol):
    """Count benzene rings with SMARTS c1ccccc1."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('c1ccccc1')
    return len(mol.GetSubstructMatches(pattern))

def count_aromatic_rings(mol):
    """Count aromatic rings from RDKit SSSR rings."""
    if not mol:
        return 0
    
    aromatic_rings = 0
    for ring in Chem.GetSSSR(mol):
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_rings += 1
    return aromatic_rings

def count_alkyl_cc(mol):
    """Count alkyl C-C single bonds with SMARTS [CX4]-[CX4]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[CX4]-[CX4]')
    return len(mol.GetSubstructMatches(pattern))

def count_alkene_cc(mol):
    """Count non-aromatic alkene C=C bonds with SMARTS [C!a]=[C!a]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[C!a]=[C!a]')
    return len(mol.GetSubstructMatches(pattern))

def count_alkyne_cc(mol):
    """Count alkyne C#C bonds with SMARTS C#C."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('C#C')
    return len(mol.GetSubstructMatches(pattern))

def count_ether(mol):
    """Count ether bonds while excluding ester and carboxylic-acid oxygens."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[C;!$(C=O)][OD2][C;!$(C=O)]')
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

def count_primary_amine(mol):
    """
    Count primary amines with SMARTS [NH2;!$(N-C=O)].
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[NH2;!$(N-C=O)]')
    return len(mol.GetSubstructMatches(pattern))

def count_secondary_amine(mol):
    """
    Count secondary amines with SMARTS [NH1;!$(N-C=O)].
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[NH1;!$(N-C=O)]')
    return len(mol.GetSubstructMatches(pattern))

def count_tertiary_amine(mol):
    """
    Count tertiary amines with SMARTS [N;H0;X3;+0;!$(N-C=O)].
    """
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[N;H0;X3;+0;!$(N-C=O)]')
    return len(mol.GetSubstructMatches(pattern))

def count_amide(mol):
    """Count amides with SMARTS C(=O)N."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('C(=O)N')
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

def count_fluoro_alkyl(mol):
    """Count fluoro-alkyl groups with SMARTS [F][CX4]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[F][CX4]')
    return len(mol.GetSubstructMatches(pattern))

def count_fluoro_aromatic(mol):
    """Count fluoro-aromatic groups with SMARTS [F]c."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[F]c')
    return len(mol.GetSubstructMatches(pattern))


def analyze_smiles_file(input_file, output_file):
    """
    Extract QM9 functional-group counts for each SMILES string in a CSV file.
    """
    functional_group_counts = []
    
    headers = [
        "Carbon_Atom_Count", "Alcohol_Hydroxyl", "Phenol_Hydroxyl", "Benzene_Ring", "Aromatic_Ring_NonBenzene", 
        "Alkene_C=C", "Alkyne_C#C", "Ether", "Aldehyde", "Ketone",
        "Carboxylic_Acid", "Ester", "Amine_Primary", "Amine_Secondary", "Amine_Tertiary",
        "Amide", "Nitrile", "Nitro", "Fluoro_Alkyl", "Fluoro_Aromatic", "Total_Count"
    ]
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        smiles_list = [row[0] for row in reader]
    
    max_total_count = 0
    max_carbon_atom_count = 0
    
    for idx, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Analyzing SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            total_count = 0
            carbon_atom_count = count_carbon_atoms(mol)
            alcohol_count = count_alcohol_hydroxyl(mol)
            phenol_count = count_phenol_hydroxyl(mol)
            benzene_count = count_benzene_rings(mol)
            aromatic_non_benzene_count = count_aromatic_rings(mol) - benzene_count
            
            alkene_cc_count = count_alkene_cc(mol)
            alkyne_cc_count = count_alkyne_cc(mol)

            ether_count = count_ether(mol)
            aldehyde_count = count_aldehyde(mol)
            ketone_count = count_ketone(mol)
            carboxylic_acid_count = count_carboxylic_acid(mol)
            ester_count = count_ester(mol)
            
            primary_amine_count = count_primary_amine(mol)
            secondary_amine_count = count_secondary_amine(mol)
            tertiary_amine_count = count_tertiary_amine(mol)
            amide_count = count_amide(mol)
            nitrile_count = count_nitrile(mol)
            nitro_count = count_nitro(mol)

            fluoro_alkyl_count = count_fluoro_alkyl(mol)
            fluoro_aromatic_count = count_fluoro_aromatic(mol)
            
            total_count = (alcohol_count + phenol_count + benzene_count + aromatic_non_benzene_count 
            + alkene_cc_count + alkyne_cc_count 
            + ether_count + aldehyde_count + ketone_count + carboxylic_acid_count + ester_count 
            + primary_amine_count + secondary_amine_count + tertiary_amine_count + amide_count + nitrile_count + nitro_count 
            + fluoro_alkyl_count + fluoro_aromatic_count)
            
            if total_count > max_total_count:
                max_total_count = total_count
            if carbon_atom_count > max_carbon_atom_count:
                max_carbon_atom_count = carbon_atom_count
            counts = [
                carbon_atom_count,
                alcohol_count, phenol_count, benzene_count, aromatic_non_benzene_count,
                alkene_cc_count, alkyne_cc_count, ether_count, aldehyde_count, ketone_count,
                carboxylic_acid_count, ester_count, primary_amine_count, secondary_amine_count,
                tertiary_amine_count, amide_count, nitrile_count, nitro_count,
                fluoro_alkyl_count, fluoro_aromatic_count, 
                total_count
            ]
            functional_group_counts.append(counts)
        else:
            functional_group_counts.append([0] * len(headers))
            print(f"Warning: failed to parse SMILES({idx}): {smiles}")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for counts in functional_group_counts:
            writer.writerow(counts)
    
    np_output_file = output_file.replace('.csv', '.npy')
    np.save(np_output_file, np.array(functional_group_counts))
    print(f"Maximum total functional-group count: {max_total_count}")
    print(f"Maximum carbon atom count: {max_carbon_atom_count}")
    return headers, functional_group_counts

HEADERS = [
        "Carbon_Atom_Count", "Alcohol_Hydroxyl", "Phenol_Hydroxyl", "Benzene_Ring", "Aromatic_Ring_NonBenzene", 
        "Alkene_C=C", "Alkyne_C#C", "Ether", "Aldehyde", "Ketone",
        "Carboxylic_Acid", "Ester", "Amine_Primary", "Amine_Secondary", "Amine_Tertiary",
        "Amide", "Nitrile", "Nitro", "Fluoro_Alkyl", "Fluoro_Aromatic", "Total_Count"
    ]

def analyze_smiles(smiles):
    """
    Extract the QM9 functional-group count vector for one SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        total_count = 0
        
        carbon_atom_count = count_carbon_atoms(mol)
        
        alcohol_count = count_alcohol_hydroxyl(mol)
        phenol_count = count_phenol_hydroxyl(mol)
        benzene_count = count_benzene_rings(mol)
        aromatic_non_benzene_count = count_aromatic_rings(mol) - benzene_count
        
        alkene_cc_count = count_alkene_cc(mol)
        alkyne_cc_count = count_alkyne_cc(mol)

        ether_count = count_ether(mol)
        aldehyde_count = count_aldehyde(mol)
        ketone_count = count_ketone(mol)
        carboxylic_acid_count = count_carboxylic_acid(mol)
        ester_count = count_ester(mol)
        
        primary_amine_count = count_primary_amine(mol)
        secondary_amine_count = count_secondary_amine(mol)
        tertiary_amine_count = count_tertiary_amine(mol)
        amide_count = count_amide(mol)
        nitrile_count = count_nitrile(mol)
        nitro_count = count_nitro(mol)

        fluoro_alkyl_count = count_fluoro_alkyl(mol)
        fluoro_aromatic_count = count_fluoro_aromatic(mol)
        
        total_count = (alcohol_count + phenol_count + benzene_count + aromatic_non_benzene_count 
        + alkene_cc_count + alkyne_cc_count 
        + ether_count + aldehyde_count + ketone_count + carboxylic_acid_count + ester_count 
        + primary_amine_count + secondary_amine_count + tertiary_amine_count + amide_count + nitrile_count + nitro_count 
        + fluoro_alkyl_count + fluoro_aromatic_count)
        
        counts = [
            carbon_atom_count, alcohol_count, phenol_count, benzene_count, aromatic_non_benzene_count,
            alkene_cc_count, alkyne_cc_count, ether_count, aldehyde_count, ketone_count,
            carboxylic_acid_count, ester_count, primary_amine_count, secondary_amine_count,
            tertiary_amine_count, amide_count, nitrile_count, nitro_count,
            fluoro_alkyl_count, fluoro_aromatic_count, total_count 
        ]
    else:
        print(f"Warning: failed to parse SMILES: {smiles}")
        
    return counts
