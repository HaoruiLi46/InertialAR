"""Stability and optional RDKit metrics for generated molecules."""

try:
    from .rdkit_functions import RDKIT_AVAILABLE, BasicMolecularMetrics

    USE_RDKIT = RDKIT_AVAILABLE
except ModuleNotFoundError:
    USE_RDKIT = False

import numpy as np
import torch

from . import bond_analyze


def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    atom_decoder = dataset_info["atom_decoder"]
    nr_bonds = np.zeros(len(positions), dtype="int")

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = float(np.sqrt(np.sum((positions[i].numpy() - positions[j].numpy()) ** 2)))
            atom1, atom2 = atom_decoder[int(atom_type[i])], atom_decoder[int(atom_type[j])]
            pair = sorted([int(atom_type[i]), int(atom_type[j])])
            if dataset_info["name"] == "qm9":
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info["name"] == "geom":
                order = bond_analyze.geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            else:
                raise ValueError(f"Unsupported dataset info: {dataset_info['name']}")
            nr_bonds[i] += order
            nr_bonds[j] += order

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[int(atom_type_i)]]
        if isinstance(possible_bonds, int):
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print(f"Invalid bonds for molecule {atom_decoder[int(atom_type_i)]} with {nr_bonds_i} bonds")
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(positions)
    return molecule_stable, nr_stable_bonds, len(positions)


def analyze_stability_for_molecules(molecule_list, dataset_info, smiles_list=None):
    one_hot = molecule_list["one_hot"]
    positions = molecule_list["x"]
    node_mask = molecule_list["node_mask"]

    atoms_per_mol = torch.sum(node_mask, dim=1) if isinstance(node_mask, torch.Tensor) else [torch.sum(m) for m in node_mask]
    n_samples = len(positions)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = positions[i].cpu().detach()
        atom_type = atom_type[: int(atoms_per_mol[i])]
        pos = pos[: int(atoms_per_mol[i])]
        processed_list.append((pos, atom_type))

    for pos, atom_type in processed_list:
        is_mol_stable, stable_bonds, atom_count = check_stability(pos, atom_type, dataset_info)
        molecule_stable += int(is_mol_stable)
        nr_stable_bonds += int(stable_bonds)
        n_atoms += int(atom_count)

    stability_dict = {
        "mol_stable": molecule_stable / float(n_samples) if n_samples else 0.0,
        "atm_stable": nr_stable_bonds / float(n_atoms) if n_atoms else 0.0,
    }

    if USE_RDKIT:
        metrics = BasicMolecularMetrics(dataset_info, smiles_list)
        return stability_dict, metrics.evaluate(processed_list)
    return stability_dict, None
