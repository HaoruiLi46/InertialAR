from rdkit import Chem
from collections import defaultdict

ALIPHATIC_RINGS = {
    'Cyclopentane': 'C1CCCC1',
    'Cyclohexane': 'C1CCCCC1',
    'Cycloheptane': 'C1CCCCCC1', # not in top500
    'Decalin': 'C1CCC2CCCCC2C1', # not in top500

    'Tetrahydrofuran': 'C1CCOC1',
    'Tetrahydropyran': 'C1CCOCC1', # not in top500
    '1,4-Dioxane': 'C1COCCO1', # not in top500
    'Piperidine': 'C1CCNCC1',
    'Pyrrolidine': 'C1CCNC1',
    'Piperazine': 'C1CNCCN1',
    'Morpholine': 'C1COCCN1',
}

ALIPHATIC_HIERARCHY = {
    'Decalin': {
        'Cyclohexane': 2 
    },
}

ALIPHATIC_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in ALIPHATIC_RINGS.items()}

def count_aliphatic_rings(mol):
    """
    Count common aliphatic rings and correct fused-ring double counting.

    Args:
        mol: RDKit molecule.

    Returns:
        Counts in ALIPHATIC_RINGS order, or zeros when mol is invalid.
    """
    if not mol: return (0,) * len(ALIPHATIC_RINGS)  

    temp_counts = defaultdict(int)
    
    for name, pattern in ALIPHATIC_PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
            
    for complex_ring, simple_rings_info in ALIPHATIC_HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring, num_to_subtract in simple_rings_info.items():
                correction = temp_counts[complex_ring] * num_to_subtract
                temp_counts[simple_ring] -= correction
    
    final_counts_list = []
    for ring_name in ALIPHATIC_RINGS:
        count = temp_counts.get(ring_name, 0)
        final_counts_list.append(max(0, count))
            
    return final_counts_list


ALIPHATIC_RINGS_v2 = {
    'Cyclopentane': 'C1CCCC1',
    'Cyclohexane_Cycloheptane': 'C1CCCCC1_C1CCCCCC1',
    'Decalin': 'C1CCC2CCCCC2C1', # not in top500

    'Tetrahydrofuran_Tetrahydropyran_Dioxane': 'C1CCOC1_C1CCOCC1_C1COCCO1',
    'Piperidine': 'C1CCNCC1',
    'Pyrrolidine': 'C1CCNC1',
    'Piperazine': 'C1CNCCN1',
    'Morpholine': 'C1COCCN1',
            }

def count_aliphatic_rings_v2(mol):
    """
    Count common aliphatic rings and correct fused-ring double counting.

    Args:
        mol: RDKit molecule.

    Returns:
        Counts in ALIPHATIC_RINGS order, or zeros when mol is invalid.
    """
    if not mol: return (0,) * len(ALIPHATIC_RINGS)  

    temp_counts = defaultdict(int)
    
    for name, pattern in ALIPHATIC_PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
            
    for complex_ring, simple_rings_info in ALIPHATIC_HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring, num_to_subtract in simple_rings_info.items():
                correction = temp_counts[complex_ring] * num_to_subtract
                temp_counts[simple_ring] -= correction
    
    counts_list = []
    for ring_name in ALIPHATIC_RINGS:
        count = temp_counts.get(ring_name, 0)
        counts_list.append(max(0, count))
    
    final_counts_list = [counts_list[0], 
                        counts_list[1]+counts_list[2], 
                        counts_list[3], 
                        counts_list[4]+counts_list[5]+counts_list[6],
                        counts_list[7], 
                        counts_list[8],
                        counts_list[9],
                        counts_list[10]
                        ]
    return final_counts_list


ALIPHATIC_RINGS_v3 = {
    'Cyclopentane': 'C1CCCC1',
    'Cyclohexane_Cycloheptane': 'C1CCCCC1_C1CCCCCC1',
    'Decalin': 'C1CCC2CCCCC2C1', # not in top500

    'Tetrahydrofuran_Tetrahydropyran_Dioxane': 'C1CCOC1_C1CCOCC1_C1COCCO1',
    'Piperidine': 'C1CCNCC1',
    'Pyrrolidine': 'C1CCNC1',
    'Piperazine': 'C1CNCCN1',
    'Morpholine': 'C1COCCN1',
            }

def count_aliphatic_rings_v3(mol):
    """
    Count common aliphatic rings and correct fused-ring double counting.

    Args:
        mol: RDKit molecule.

    Returns:
        Counts in ALIPHATIC_RINGS order, or zeros when mol is invalid.
    """
    if not mol: return (0,) * len(ALIPHATIC_RINGS)  

    temp_counts = defaultdict(int)
    
    for name, pattern in ALIPHATIC_PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
            
    for complex_ring, simple_rings_info in ALIPHATIC_HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring, num_to_subtract in simple_rings_info.items():
                correction = temp_counts[complex_ring] * num_to_subtract
                temp_counts[simple_ring] -= correction
    
    counts_list = []
    for ring_name in ALIPHATIC_RINGS:
        count = temp_counts.get(ring_name, 0)
        counts_list.append(max(0, count))
    
    final_counts_list = [counts_list[0], 
                        counts_list[1]+counts_list[2], 
                        counts_list[3], 
                        counts_list[4]+counts_list[5]+counts_list[6],
                        counts_list[7], 
                        counts_list[8],
                        counts_list[9],
                        counts_list[10]
                        ]
    return final_counts_list


ALIPHATIC_RINGS_v4 = {
    'Cyclopentane': 'C1CCCC1',
    'Cyclohexane_Cycloheptane': 'C1CCCCC1_C1CCCCCC1',
    'Tetrahydrofuran_Tetrahydropyran_Dioxane': 'C1CCOC1_C1CCOCC1_C1COCCO1',
    'Piperidine': 'C1CCNCC1',
    'Pyrrolidine': 'C1CCNC1',
    'Piperazine': 'C1CNCCN1',
    'Morpholine': 'C1COCCN1',
    'Decalin': 'C1CCC2CCCCC2C1', # not in top500
            }

def count_aliphatic_rings_v4(mol):
    """
    Count common aliphatic rings and correct fused-ring double counting.

    Args:
        mol: RDKit molecule.

    Returns:
        Counts in ALIPHATIC_RINGS order, or zeros when mol is invalid.
    """
    if not mol: return (0,) * len(ALIPHATIC_RINGS)  

    temp_counts = defaultdict(int)
    
    for name, pattern in ALIPHATIC_PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
            
    for complex_ring, simple_rings_info in ALIPHATIC_HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring, num_to_subtract in simple_rings_info.items():
                correction = temp_counts[complex_ring] * num_to_subtract
                temp_counts[simple_ring] -= correction
    
    counts_list = []
    for ring_name in ALIPHATIC_RINGS:
        count = temp_counts.get(ring_name, 0)
        counts_list.append(max(0, count))
    
    final_counts_list = [counts_list[0], 
                        counts_list[1]+counts_list[2],  
                        counts_list[4]+counts_list[5]+counts_list[6],
                        counts_list[7], 
                        counts_list[8],
                        counts_list[9],
                        counts_list[10],
                        counts_list[3],
                        ]
    
    for i in range(len(final_counts_list) - 1, -1, -1):
        if final_counts_list[i] > 0:
            for j in range(i):
                final_counts_list[j] = 0
            break
    else:
        pass

    return final_counts_list

ALIPHATIC_RINGS_v8 = {
    'Aliphatic_carbocycles': 'Aliphatic_carbocycles',

    'Aliphatic heterocycles': 'Aliphatic heterocycles',
            }

def count_aliphatic_rings_v8(mol):
    """
    Count aliphatic carbocycles and heterocycles with the selected public featurizer.

    Args:
        mol: RDKit molecule.

    Returns:
        Two counts: aliphatic carbocycles and aliphatic heterocycles.
    """
    if not mol: return (0,) * len(ALIPHATIC_RINGS)  

    temp_counts = defaultdict(int)
    
    for name, pattern in ALIPHATIC_PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            temp_counts[name] = len(matches)
            
    for complex_ring, simple_rings_info in ALIPHATIC_HIERARCHY.items():
        if temp_counts[complex_ring] > 0:
            for simple_ring, num_to_subtract in simple_rings_info.items():
                correction = temp_counts[complex_ring] * num_to_subtract
                temp_counts[simple_ring] -= correction
    
    counts_list = []
    for ring_name in ALIPHATIC_RINGS:
        count = temp_counts.get(ring_name, 0)
        counts_list.append(max(0, count))
    
    carbocycles_counts = sum(counts_list[:3])
    heterocycles_counts = sum(counts_list[3:])
    
    final_counts_list = [
                        carbocycles_counts,  
                        heterocycles_counts,
                        ]

    return final_counts_list
