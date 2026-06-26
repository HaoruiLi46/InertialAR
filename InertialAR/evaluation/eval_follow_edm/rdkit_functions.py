"""RDKit validity and uniqueness helpers for generated 3D molecules."""

try:
    from rdkit import Chem
except ImportError:
    Chem = None

import torch

from .bond_analyze import geom_predictor, get_bond_order

RDKIT_AVAILABLE = Chem is not None

bond_dict = [None]
if RDKIT_AVAILABLE:
    bond_dict += [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]


class BasicMolecularMetrics:
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_info = dataset_info
        self.dataset_smiles_list = dataset_smiles_list

    def compute_validity(self, generated):
        if not RDKIT_AVAILABLE or len(generated) == 0:
            return [], 0.0, []

        valid = []
        log = []
        for graph in generated:
            mol = build_molecule(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                except Exception:
                    smiles = None

            if smiles is not None:
                valid.append(smiles)
                log.append(smiles)
            else:
                log.append("None")

        return valid, len(valid) / len(generated), log

    @staticmethod
    def compute_uniqueness(valid):
        if len(valid) == 0:
            return [], 0.0
        unique = sorted(set(valid))
        return unique, len(unique) / len(valid)

    def evaluate(self, generated):
        valid, validity, log = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")
        else:
            unique = []
            uniqueness = 0.0
        return [validity, uniqueness], unique, log


def mol2smiles(mol):
    if mol is None or Chem is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def build_molecule(positions, atom_types, dataset_info):
    if Chem is None:
        return None

    atom_decoder = dataset_info["atom_decoder"]
    atom_numbers, adjacency, edge_types = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()

    for atom in atom_numbers:
        mol.AddAtom(Chem.Atom(atom_decoder[int(atom)]))

    all_bonds = torch.nonzero(adjacency)
    for bond in all_bonds:
        mol.AddBond(int(bond[0]), int(bond[1]), bond_dict[int(edge_types[bond[0], bond[1]])])
    mol.UpdatePropertyCache(strict=False)
    return mol.GetMol()


def build_xae_molecule(positions, atom_types, dataset_info):
    atom_decoder = dataset_info["atom_decoder"]
    n_atoms = positions.shape[0]
    atom_numbers = atom_types
    adjacency = torch.zeros((n_atoms, n_atoms), dtype=torch.bool)
    edge_types = torch.zeros((n_atoms, n_atoms), dtype=torch.int)

    dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0), p=2).squeeze(0)
    for i in range(n_atoms):
        for j in range(i):
            pair = sorted([int(atom_types[i]), int(atom_types[j])])
            if dataset_info["name"] == "qm9":
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info["name"] == "geom":
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            else:
                raise ValueError(f"Unsupported dataset info: {dataset_info['name']}")
            if order > 0:
                adjacency[i, j] = 1
                edge_types[i, j] = order
    return atom_numbers, adjacency, edge_types
