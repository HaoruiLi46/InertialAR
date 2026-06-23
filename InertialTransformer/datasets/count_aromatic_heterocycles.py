from rdkit import Chem
from collections import defaultdict

AROMATIC_HETEROCYCLES_EXTENDED = {
    'Furan': 'c1ccoc1',
    'Thiophene': 'c1ccsc1',
    'Pyrazole': 'n1nccc1',
    'Oxazole': 'o1cncc1',
    'Isoxazole': 'o1nccc1',
    'Thiazole': 'c1cncs1',
    '1,3,4-Thiadiazole': 's1cnnc1',
    'Imidazole': 'n1cncc1', #No in top100
    '1,2,3-Triazole': 'c1nnnc1', #No in top100
    '1,2,4-Triazole': 'c1ncnn1',
    'Tetrazole': 'c1nnnn1', #No in top100
    '1,3,4-Oxadiazole': 'c1nnco1',
    '1,2,4-Oxadiazole': 'c1nocn1',
    'Pyrrole': 'n1cccc1', #*No in top200 top243

    'Pyridine': 'c1cnccc1',
    'Pyrimidine': 'c1cncnc1',
    'Pyridazine': 'c1ccnnc1',  #No in top100
    'Pyrazine': 'c1cnccn1', #No in top200 top238

    'Indole': 'c1ccc2nccc2c1',
    'Quinoline': 'c1ccc2ncccc2c1',
    'Benzimidazole': 'c1ccc2ncnc2c1',
    'Coumarin': 'c1ccc2ccc(=O)oc2c1',
    'Isoquinoline': 'c1ccc2ccncc2c1', #No in top100
    'Benzofuran': 'c1ccc2occc2c1', #No in top100
    'Benzothiazole': 'c1ccc2ncsc2c1', #No in top100
    'Indazole': 'c1ccc2nncc2c1', #No in top200 top348
    'Purine': 'c1c2c(ncn2)ncn1', #No in top500
    'Benzoxazole': 'c1ccc2ncoc2c1', #No in top200 top415
    'Benzisoxazole': 'c1ccc2oncc2c1', #No in top500
    'Benzothiophene': 'c1ccc2sccc2c1', #No in top500
}

HIERARCHY = {
    'Indole': ['Pyrrole'],
    'Quinoline': ['Pyridine'],
    'Isoquinoline': ['Pyridine'],
    'Benzofuran': ['Furan'],
    'Benzimidazole': ['Imidazole'],
    'Indazole': ['Pyrazole'],
    'Benzothiazole': ['Thiazole'],
    'Purine': ['Imidazole', 'Pyrimidine'],
    'Benzoxazole': ['Oxazole'],
    'Benzisoxazole': ['Isoxazole'],
    'Benzothiophene': ['Thiophene'],
}


HETEROATOM_COUNTS = {
    'Pyrrole': {'N': 1, 'O': 0, 'S': 0},
    'Furan': {'N': 0, 'O': 1, 'S': 0},
    'Thiophene': {'N': 0, 'O': 0, 'S': 1},
    'Imidazole': {'N': 2, 'O': 0, 'S': 0},
    'Pyrazole': {'N': 2, 'O': 0, 'S': 0},
    'Oxazole': {'N': 1, 'O': 1, 'S': 0},
    'Isoxazole': {'N': 1, 'O': 1, 'S': 0},
    'Thiazole': {'N': 1, 'O': 0, 'S': 1},
    '1,2,3-Triazole': {'N': 3, 'O': 0, 'S': 0},
    '1,2,4-Triazole': {'N': 3, 'O': 0, 'S': 0},
    'Tetrazole': {'N': 4, 'O': 0, 'S': 0},
    '1,3,4-Oxadiazole': {'N': 2, 'O': 1, 'S': 0},
    '1,2,4-Oxadiazole': {'N': 2, 'O': 1, 'S': 0},
    '1,3,4-Thiadiazole': {'N': 2, 'O': 0, 'S': 1},

    'Pyridine': {'N': 1, 'O': 0, 'S': 0},
    'Pyrimidine': {'N': 2, 'O': 0, 'S': 0},
    'Pyridazine': {'N': 2, 'O': 0, 'S': 0},
    'Pyrazine': {'N': 2, 'O': 0, 'S': 0},

    'Indole': {'N': 1, 'O': 0, 'S': 0},
    'Quinoline': {'N': 1, 'O': 0, 'S': 0},
    'Isoquinoline': {'N': 1, 'O': 0, 'S': 0},
    'Benzofuran': {'N': 0, 'O': 1, 'S': 0},
    'Benzimidazole': {'N': 2, 'O': 0, 'S': 0},
    'Indazole': {'N': 2, 'O': 0, 'S': 0},
    'Benzothiazole': {'N': 1, 'O': 0, 'S': 1},
    'Coumarin': {'N': 0, 'O': 2, 'S': 0},
    'Purine': {'N': 4, 'O': 0, 'S': 0},
    'Benzoxazole': {'N': 1, 'O': 1, 'S': 0},
    'Benzisoxazole': {'N': 1, 'O': 1, 'S': 0},
    'Benzothiophene': {'N': 0, 'O': 0, 'S': 1},
}

PATTERNS_EXTENDED = {name: Chem.MolFromSmarts(smarts) for name, smarts in AROMATIC_HETEROCYCLES_EXTENDED.items()}

def count_aromatic_heterocycles_extended(mol):
    """
    Count aromatic heterocycles with hierarchy correction and heteroatom totals.

    Args:
        mol: RDKit molecule.

    Returns:
        tuple: (final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count)
    """
    if not mol:
        return (0,) * len(AROMATIC_HETEROCYCLES_EXTENDED), 0, 0, 0, 0

    temp_counts = defaultdict(int)
    
    for name, pattern in PATTERNS_EXTENDED.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
    
    for complex_ring, simple_rings in HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring in simple_rings:
                temp_counts[simple_ring] -= temp_counts[complex_ring]
    
    final_counts_list = []
    sum_count = 0
    n_atom_count = 0
    o_atom_count = 0
    s_atom_count = 0
    
    for ring_name in AROMATIC_HETEROCYCLES_EXTENDED:
        count = temp_counts.get(ring_name, 0)
        adjusted_count = max(0, count)
        final_counts_list.append(adjusted_count)
        sum_count += adjusted_count
        
        if adjusted_count > 0:
            heteroatom_info = HETEROATOM_COUNTS[ring_name]
            n_atom_count += adjusted_count * heteroatom_info['N']
            o_atom_count += adjusted_count * heteroatom_info['O']
            s_atom_count += adjusted_count * heteroatom_info['S']
    
    return final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count

AROMATIC_HETEROCYCLES_EXTENDED_v2 = {
    'Furan': 'c1ccoc1',
    'Thiophene': 'c1ccsc1',
    'Pyrazole': 'n1nccc1',
    'Oxazole': 'o1cncc1',
    'Isoxazole': 'o1nccc1',
    'Thiazole': 'c1cncs1',
    '1,3,4-Thiadiazole': 's1cnnc1',
    'Imidazole': 'n1cncc1', #No in top100
    '1,2,3-Triazole': 'c1nnnc1', #No in top100
    '1,2,4-Triazole': 'c1ncnn1',
    'Tetrazole': 'c1nnnn1', #No in top100
    '1,3,4-Oxadiazole': 'c1nnco1',
    '1,2,4-Oxadiazole': 'c1nocn1',
    'Pyrrole': 'n1cccc1', #*No in top200 top243

    'Pyridine': 'c1cnccc1',
    'Pyrimidine': 'c1cncnc1',
    'Pyridazine': 'c1ccnnc1',  #No in top100
    'Pyrazine': 'c1cnccn1', #No in top200 top238

    'Indole': 'c1ccc2nccc2c1',
    'Quinoline': 'c1ccc2ncccc2c1',
    'Benzimidazole': 'c1ccc2ncnc2c1',
    'Coumarin': 'c1ccc2ccc(=O)oc2c1',
    'Isoquinoline': 'c1ccc2ccncc2c1', #No in top100
    'Benzofuran': 'c1ccc2occc2c1', #No in top100
    'Benzothiazole': 'c1ccc2ncsc2c1', #No in top100
    'Indazole': 'c1ccc2nncc2c1', #No in top200 top348
    'Benzoxazole_Benzisoxazole': 'c1ccc2ncoc2c1_c1ccc2oncc2c1', #No in top200 top415
    'Purine_Benzothiophene': 'c1c2c(ncn2)ncn1_c1ccc2sccc2c1', #No in top500
}

def count_aromatic_heterocycles_extended_v2(mol):
    """
    Count aromatic heterocycles with hierarchy correction and heteroatom totals.

    Args:
        mol: RDKit molecule.

    Returns:
        tuple: (final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count)
    """
    if not mol:
        return (0,) * len(AROMATIC_HETEROCYCLES_EXTENDED), 0, 0, 0, 0

    temp_counts = defaultdict(int)
    
    for name, pattern in PATTERNS_EXTENDED.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
    
    for complex_ring, simple_rings in HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring in simple_rings:
                temp_counts[simple_ring] -= temp_counts[complex_ring]
    
    counts_list = []
    sum_count = 0
    n_atom_count = 0
    o_atom_count = 0
    s_atom_count = 0
    
    for ring_name in AROMATIC_HETEROCYCLES_EXTENDED:
        count = temp_counts.get(ring_name, 0)
        adjusted_count = max(0, count)
        counts_list.append(adjusted_count)
        sum_count += adjusted_count
        
        if adjusted_count > 0:
            heteroatom_info = HETEROATOM_COUNTS[ring_name]
            n_atom_count += adjusted_count * heteroatom_info['N']
            o_atom_count += adjusted_count * heteroatom_info['O']
            s_atom_count += adjusted_count * heteroatom_info['S']
            
    final_counts_list = []
    final_counts_list.extend(counts_list[:-4])
    final_counts_list.append(counts_list[-2]+counts_list[-3])
    final_counts_list.append(counts_list[-4]+counts_list[-1])
    
    return final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count


AROMATIC_HETEROCYCLES_EXTENDED_v3 = {
    'Furan': 'c1ccoc1',
    'Thiophene': 'c1ccsc1',
    'Pyrazole': 'n1nccc1',
    'Oxazole': 'o1cncc1',
    'Isoxazole': 'o1nccc1',
    'Thiazole': 'c1cncs1',
    'Imidazole': 'n1cncc1', #No in top100
    '1,3,4-Thiadiazole': 's1cnnc1',
    'Triazole': 'c1nnnc1_c1ncnn1', #1,2,4-Triazole 1,2,3-Triazole No in top100
    'Tetrazole': 'c1nnnn1', #No in top100
    'Oxadiazole': 'c1nnco1_c1nocn1',
    'Pyrrole': 'n1cccc1', #*No in top200 top243
    'Pyridine': 'c1cnccc1',
    'Pyrimidine': 'c1cncnc1',
    'Pyridazine': 'c1ccnnc1',  #No in top100
    'Pyrazine': 'c1cnccn1', #No in top200 top238

    'Indole': 'c1ccc2nccc2c1',
    'Quinoline': 'c1ccc2ncccc2c1',
    'Benzimidazole': 'c1ccc2ncnc2c1',
    'Coumarin': 'c1ccc2ccc(=O)oc2c1',
    'Isoquinoline': 'c1ccc2ccncc2c1', #No in top100
    'Benzofuran': 'c1ccc2occc2c1', #No in top100
    'Benzothiazole': 'c1ccc2ncsc2c1', #No in top100
    'Indazole': 'c1ccc2nncc2c1', #No in top200 top348
    'Benzoxazole_Benzisoxazole_Purine_Benzothiophene': 'c1ccc2ncoc2c1_c1c2c(ncn2)ncn1_c1ccc2oncc2c1_c1ccc2sccc2c1', #No in top500
}

def count_aromatic_heterocycles_extended_v3(mol):
    """
    Count aromatic heterocycles with hierarchy correction and heteroatom totals.

    Args:
        mol: RDKit molecule.

    Returns:
        tuple: (final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count)
    """
    if not mol:
        return (0,) * len(AROMATIC_HETEROCYCLES_EXTENDED), 0, 0, 0, 0

    temp_counts = defaultdict(int)
    
    for name, pattern in PATTERNS_EXTENDED.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
    
    for complex_ring, simple_rings in HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring in simple_rings:
                temp_counts[simple_ring] -= temp_counts[complex_ring]
    
    counts_list = []
    sum_count = 0
    n_atom_count = 0
    o_atom_count = 0
    s_atom_count = 0
    
    for ring_name in AROMATIC_HETEROCYCLES_EXTENDED:
        count = temp_counts.get(ring_name, 0)
        adjusted_count = max(0, count)
        counts_list.append(adjusted_count)
        sum_count += adjusted_count
        
        if adjusted_count > 0:
            heteroatom_info = HETEROATOM_COUNTS[ring_name]
            n_atom_count += adjusted_count * heteroatom_info['N']
            o_atom_count += adjusted_count * heteroatom_info['O']
            s_atom_count += adjusted_count * heteroatom_info['S']
            
    final_counts_list = []
    final_counts_list.extend(counts_list[:8])
    final_counts_list.append(counts_list[8]+counts_list[9])
    final_counts_list.append(counts_list[10])
    final_counts_list.append(counts_list[11]+counts_list[12])
    
    final_counts_list.extend(counts_list[13:-4])
    
    final_counts_list.append(counts_list[-4]+counts_list[-3]+counts_list[-2]+counts_list[-1])
    
    
    return final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count

AROMATIC_HETEROCYCLES_EXTENDED_v4 = {
    'Furan': 'c1ccoc1',
    'Thiophene': 'c1ccsc1',
    'Pyrazole': 'n1nccc1',
    'Oxazole': 'o1cncc1',
    'Isoxazole': 'o1nccc1',
    'Thiazole': 'c1cncs1',
    'Imidazole': 'n1cncc1', #No in top100
    '1,3,4-Thiadiazole': 's1cnnc1',
    'Triazole': 'c1nnnc1_c1ncnn1', #1,2,4-Triazole 1,2,3-Triazole No in top100
    'Tetrazole': 'c1nnnn1', #No in top100
    'Oxadiazole': 'c1nnco1_c1nocn1',
    'Pyrrole': 'n1cccc1', #*No in top200 top243

    'Pyridine': 'c1cnccc1',
    'Pyrimidine': 'c1cncnc1',
    'Pyridazine': 'c1ccnnc1',  #No in top100
    'Pyrazine': 'c1cnccn1', #No in top200 top238

    'Indole': 'c1ccc2nccc2c1',
    'Quinoline': 'c1ccc2ncccc2c1',
    'Benzimidazole': 'c1ccc2ncnc2c1',
    'Coumarin': 'c1ccc2ccc(=O)oc2c1',
    'Isoquinoline': 'c1ccc2ccncc2c1', #No in top100
    'Benzofuran': 'c1ccc2occc2c1', #No in top100
    'Benzothiazole': 'c1ccc2ncsc2c1', #No in top100
    'Indazole': 'c1ccc2nncc2c1', #No in top200 top348
    'Benzoxazole_Benzisoxazole_Purine_Benzothiophene': 'c1ccc2ncoc2c1_c1c2c(ncn2)ncn1_c1ccc2oncc2c1_c1ccc2sccc2c1', #No in top500
}

def count_aromatic_heterocycles_extended_v4(mol):
    """
    Count aromatic heterocycles with hierarchy correction and heteroatom totals.

    Args:
        mol: RDKit molecule.

    Returns:
        tuple: (final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count)
    """
    if not mol:
        return (0,) * len(AROMATIC_HETEROCYCLES_EXTENDED), 0, 0, 0, 0

    temp_counts = defaultdict(int)
    
    for name, pattern in PATTERNS_EXTENDED.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
    
    for complex_ring, simple_rings in HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring in simple_rings:
                temp_counts[simple_ring] -= temp_counts[complex_ring]
    
    counts_list = []
    sum_count = 0
    n_atom_count = 0
    o_atom_count = 0
    s_atom_count = 0
    
    for ring_name in AROMATIC_HETEROCYCLES_EXTENDED:
        count = temp_counts.get(ring_name, 0)
        adjusted_count = max(0, count)
        counts_list.append(adjusted_count)
        sum_count += adjusted_count
        
        if adjusted_count > 0:
            heteroatom_info = HETEROATOM_COUNTS[ring_name]
            n_atom_count += adjusted_count * heteroatom_info['N']
            o_atom_count += adjusted_count * heteroatom_info['O']
            s_atom_count += adjusted_count * heteroatom_info['S']
            
    final_counts_list = []
    final_counts_list.extend(counts_list[:8])
    final_counts_list.append(counts_list[8]+counts_list[9])
    final_counts_list.append(counts_list[10])
    final_counts_list.append(counts_list[11]+counts_list[12])

    final_counts_list.extend(counts_list[13:-4])

    final_counts_list.append(counts_list[-4]+counts_list[-3]+counts_list[-2]+counts_list[-1])

    for i in range(len(final_counts_list) - 1, -1, -1):
        if final_counts_list[i] > 0:
            for j in range(i):
                final_counts_list[j] = 0
            break
    else:
        pass

    return final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count


AROMATIC_HETEROCYCLES_EXTENDED_v5 = {
    'Furan': 'c1ccoc1',
    'Thiophene': 'c1ccsc1',
    'Pyrazole': 'n1nccc1',
    'Oxazole': 'o1cncc1',
    'Isoxazole': 'o1nccc1',
    'Thiazole': 'c1cncs1',
    'Imidazole': 'n1cncc1', #No in top100
    '1,3,4-Thiadiazole': 's1cnnc1',
    'Triazole': 'c1nnnc1_c1ncnn1', #1,2,4-Triazole 1,2,3-Triazole No in top100
    'Tetrazole': 'c1nnnn1', #No in top100
    'Oxadiazole': 'c1nnco1_c1nocn1',
    'Pyrrole': 'n1cccc1', #*No in top200 top243

    'Pyridine': 'c1cnccc1',
    'Pyrimidine': 'c1cncnc1',
    'Pyridazine': 'c1ccnnc1',  #No in top100
    'Pyrazine': 'c1cnccn1', #No in top200 top238

    'Indole': 'c1ccc2nccc2c1',
    'Quinoline': 'c1ccc2ncccc2c1',
    'Benzimidazole': 'c1ccc2ncnc2c1',
    'Coumarin': 'c1ccc2ccc(=O)oc2c1',
    'Isoquinoline': 'c1ccc2ccncc2c1', #No in top100
    'Benzofuran': 'c1ccc2occc2c1', #No in top100
    'Benzothiazole': 'c1ccc2ncsc2c1', #No in top100
    'Indazole': 'c1ccc2nncc2c1', #No in top200 top348
    'Benzoxazole_Benzisoxazole': 'c1ccc2ncoc2c1_c1c2c(ncn2)ncn1',
    'Purine_Benzothiophene': 'c1ccc2oncc2c1_c1ccc2sccc2c1', #No in top500
}

def count_aromatic_heterocycles_extended_v5(mol):
    """
    Count aromatic heterocycles with hierarchy correction and heteroatom totals.

    Args:
        mol: RDKit molecule.

    Returns:
        tuple: (final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count)
    """
    if not mol:
        return (0,) * len(AROMATIC_HETEROCYCLES_EXTENDED), 0, 0, 0, 0

    temp_counts = defaultdict(int)
    
    for name, pattern in PATTERNS_EXTENDED.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
    
    for complex_ring, simple_rings in HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring in simple_rings:
                temp_counts[simple_ring] -= temp_counts[complex_ring]
    
    counts_list = []
    sum_count = 0
    n_atom_count = 0
    o_atom_count = 0
    s_atom_count = 0
    
    for ring_name in AROMATIC_HETEROCYCLES_EXTENDED:
        count = temp_counts.get(ring_name, 0)
        adjusted_count = max(0, count)
        counts_list.append(adjusted_count)
        sum_count += adjusted_count
        
        if adjusted_count > 0:
            heteroatom_info = HETEROATOM_COUNTS[ring_name]
            n_atom_count += adjusted_count * heteroatom_info['N']
            o_atom_count += adjusted_count * heteroatom_info['O']
            s_atom_count += adjusted_count * heteroatom_info['S']
            
    final_counts_list = []
    final_counts_list.extend(counts_list[:8])
    final_counts_list.append(counts_list[8]+counts_list[9])
    final_counts_list.append(counts_list[10])
    final_counts_list.append(counts_list[11]+counts_list[12])

    final_counts_list.extend(counts_list[13:-4])

    final_counts_list.append(counts_list[-2]+counts_list[-3])
    final_counts_list.append(counts_list[-4]+counts_list[-1])

    for i in range(len(final_counts_list) - 1, -1, -1):
        if final_counts_list[i] > 0:
            for j in range(i):
                final_counts_list[j] = 0
            break
    else:
        pass

    return final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count
AROMATIC_HETEROCYCLES_EXTENDED_v8 = {
    'five_aromatic_ring': 'FIVE_AROMATIC_RING',
    'six_aromatic_ring': 'SIX_AROMATIC_RING',
    'complex_aromatic_ring': 'COMPLEX_AROMATIC_RING',
}

def count_aromatic_heterocycles_extended_v8(mol):
    """
    Count aromatic heterocycles grouped into five-, six-, and fused-ring bins.

    Args:
        mol: RDKit molecule.

    Returns:
        tuple: (final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count)
    """
    if not mol:
        return (0,) * len(AROMATIC_HETEROCYCLES_EXTENDED), 0, 0, 0, 0

    temp_counts = defaultdict(int)
    
    for name, pattern in PATTERNS_EXTENDED.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
    
    for complex_ring, simple_rings in HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring in simple_rings:
                temp_counts[simple_ring] -= temp_counts[complex_ring]
    
    counts_list = []
    sum_count = 0
    n_atom_count = 0
    o_atom_count = 0
    s_atom_count = 0
    
    for ring_name in AROMATIC_HETEROCYCLES_EXTENDED:
        count = temp_counts.get(ring_name, 0)
        adjusted_count = max(0, count)
        counts_list.append(adjusted_count)
        sum_count += adjusted_count
        
        if adjusted_count > 0:
            heteroatom_info = HETEROATOM_COUNTS[ring_name]
            n_atom_count += adjusted_count * heteroatom_info['N']
            o_atom_count += adjusted_count * heteroatom_info['O']
            s_atom_count += adjusted_count * heteroatom_info['S']
            
    final_counts_list = []
    five_ring_counts = sum(counts_list[:14])
    six_ring_counts = sum(counts_list[14:18])
    complex_ring_counts = sum(counts_list[18:])


    final_counts_list.append(five_ring_counts)
    final_counts_list.append(six_ring_counts)
    final_counts_list.append(complex_ring_counts)
    
    
    return final_counts_list, sum_count, n_atom_count, o_atom_count, s_atom_count
