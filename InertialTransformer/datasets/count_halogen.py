from rdkit import Chem
from collections import defaultdict

def count_fluoro_alkyl(mol):
    """Count fluoro-alkyl groups with SMARTS [F][CX4]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[F][CX4]')
    return len(mol.GetSubstructMatches(pattern))

def count_fluoro_aromatic(mol):
    """Count fluoro-aromatic groups with SMARTS [F]-a."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[F]-a')
    return len(mol.GetSubstructMatches(pattern))

def count_chloro_alkyl(mol):
    """Count chloro-alkyl groups with SMARTS [Cl][CX4]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[Cl][CX4]')
    return len(mol.GetSubstructMatches(pattern))

def count_chloro_aromatic(mol):
    """Count chloro-aromatic groups with SMARTS [Cl]-a."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[Cl]-a')
    return len(mol.GetSubstructMatches(pattern))

def count_bromo_alkyl(mol):
    """Count bromo-alkyl groups with SMARTS [Br][CX4]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[Br][CX4]')
    return len(mol.GetSubstructMatches(pattern))

def count_bromo_aromatic(mol):
    """Count bromo-aromatic groups with SMARTS [Br]-a."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[Br]-a')
    return len(mol.GetSubstructMatches(pattern))

def count_iodo_alkyl(mol):
    """Count iodo-alkyl groups with SMARTS [I][CX4]."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[I][CX4]')
    return len(mol.GetSubstructMatches(pattern))

def count_iodo_aromatic(mol):
    """Count iodo-aromatic groups with SMARTS [I]-a."""
    if not mol: return 0
    pattern = Chem.MolFromSmarts('[I]-a')
    return len(mol.GetSubstructMatches(pattern))


HALOGEN_GROUP_HEADERS = [
    'fluoro_alkyl',
    'fluoro_aromatic',
    'chloro_alkyl',
    'chloro_aromatic',
    'bromo_alkyl',
    'bromo_aromatic',
    'iodo_alkyl',
    'iodo_aromatic',
]

def count_halogen(mol):
    fluoro_alkyl = count_fluoro_alkyl(mol)
    fluoro_aromatic = count_fluoro_aromatic(mol)
    chloro_alkyl = count_chloro_alkyl(mol)
    chloro_aromatic = count_chloro_aromatic(mol)
    bromo_alkyl = count_bromo_alkyl(mol)
    bromo_aromatic = count_bromo_aromatic(mol)
    iodo_alkyl = count_iodo_alkyl(mol)
    iodo_aromatic = count_iodo_aromatic(mol)
    halogen_functional_group_counts = [
        fluoro_alkyl,
        fluoro_aromatic,
        chloro_alkyl,
        chloro_aromatic,
        bromo_alkyl,
        bromo_aromatic,
        iodo_alkyl,
        iodo_aromatic,
    ]
    fluoro_atom_count = fluoro_alkyl + fluoro_aromatic
    chloro_atom_count = chloro_alkyl + chloro_aromatic
    bromo_atom_count = bromo_alkyl + bromo_aromatic
    iodo_atom_count = iodo_alkyl + iodo_aromatic
    return halogen_functional_group_counts, fluoro_atom_count, chloro_atom_count, bromo_atom_count, iodo_atom_count

HALOGEN_GROUP_HEADERS_v2 = [
    'fluoro_alkyl',
    'fluoro_aromatic',
    'chloro_alkyl',
    'chloro_aromatic',
    'bromo_alkyl',
    'bromo_aromatic',
    'iodo_alkyl',
    'iodo_aromatic',
]

def count_halogen_v2(mol):
    fluoro_alkyl = count_fluoro_alkyl(mol)
    fluoro_aromatic = count_fluoro_aromatic(mol)
    chloro_alkyl = count_chloro_alkyl(mol)
    chloro_aromatic = count_chloro_aromatic(mol)
    bromo_alkyl = count_bromo_alkyl(mol)
    bromo_aromatic = count_bromo_aromatic(mol)
    iodo_alkyl = count_iodo_alkyl(mol)
    iodo_aromatic = count_iodo_aromatic(mol)
    halogen_functional_group_counts = [
        fluoro_alkyl,
        fluoro_aromatic,
        chloro_alkyl,
        chloro_aromatic,
        bromo_alkyl,
        bromo_aromatic,
        iodo_alkyl,
        iodo_aromatic,
    ]
    fluoro_atom_count = fluoro_alkyl + fluoro_aromatic
    chloro_atom_count = chloro_alkyl + chloro_aromatic
    bromo_atom_count = bromo_alkyl + bromo_aromatic
    iodo_atom_count = iodo_alkyl + iodo_aromatic
    return halogen_functional_group_counts, fluoro_atom_count, chloro_atom_count, bromo_atom_count, iodo_atom_count


HALOGEN_GROUP_HEADERS_v3 = [
    'fluoro_alkyl_aromatic',
    'chloro_alkyl_aromatic',
    'bromo_alkyl_aromatic',
    'iodo_alkyl_aromatic',
]

def count_halogen_v3(mol):
    fluoro_alkyl = count_fluoro_alkyl(mol)
    fluoro_aromatic = count_fluoro_aromatic(mol)
    chloro_alkyl = count_chloro_alkyl(mol)
    chloro_aromatic = count_chloro_aromatic(mol)
    bromo_alkyl = count_bromo_alkyl(mol)
    bromo_aromatic = count_bromo_aromatic(mol)
    iodo_alkyl = count_iodo_alkyl(mol)
    iodo_aromatic = count_iodo_aromatic(mol)
    halogen_functional_group_counts = [
        fluoro_alkyl+fluoro_aromatic,
        chloro_alkyl+chloro_aromatic,
        bromo_alkyl+bromo_aromatic,
        iodo_alkyl+iodo_aromatic,
    ]
    fluoro_atom_count = fluoro_alkyl + fluoro_aromatic
    chloro_atom_count = chloro_alkyl + chloro_aromatic
    bromo_atom_count = bromo_alkyl + bromo_aromatic
    iodo_atom_count = iodo_alkyl + iodo_aromatic
    return halogen_functional_group_counts, fluoro_atom_count, chloro_atom_count, bromo_atom_count, iodo_atom_count


HALOGEN_GROUP_HEADERS_v4 = [
    'fluoro_alkyl_aromatic',
    'chloro_alkyl_aromatic',
    'bromo_alkyl_aromatic',
    'iodo_alkyl_aromatic',
]

def count_halogen_v4(mol):
    fluoro_alkyl = count_fluoro_alkyl(mol)
    fluoro_aromatic = count_fluoro_aromatic(mol)
    chloro_alkyl = count_chloro_alkyl(mol)
    chloro_aromatic = count_chloro_aromatic(mol)
    bromo_alkyl = count_bromo_alkyl(mol)
    bromo_aromatic = count_bromo_aromatic(mol)
    iodo_alkyl = count_iodo_alkyl(mol)
    iodo_aromatic = count_iodo_aromatic(mol)
    halogen_functional_group_counts = [
        fluoro_alkyl+fluoro_aromatic,
        chloro_alkyl+chloro_aromatic,
        bromo_alkyl+bromo_aromatic,
        iodo_alkyl+iodo_aromatic,
    ]
    fluoro_atom_count = fluoro_alkyl + fluoro_aromatic
    chloro_atom_count = chloro_alkyl + chloro_aromatic
    bromo_atom_count = bromo_alkyl + bromo_aromatic
    iodo_atom_count = iodo_alkyl + iodo_aromatic
    return halogen_functional_group_counts, fluoro_atom_count, chloro_atom_count, bromo_atom_count, iodo_atom_count
