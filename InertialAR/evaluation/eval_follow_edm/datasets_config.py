"""Dataset metadata for EDM-style generated-molecule evaluation."""


QM9_WITH_H = {
    "name": "qm9",
    "atom_encoder": {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4},
    "atom_decoder": ["H", "C", "N", "O", "F"],
    "max_n_nodes": 29,
    "with_h": True,
}

QM9_WITHOUT_H = {
    "name": "qm9",
    "atom_encoder": {"C": 0, "N": 1, "O": 2, "F": 3},
    "atom_decoder": ["C", "N", "O", "F"],
    "max_n_nodes": 29,
    "with_h": False,
}

GEOM_WITH_H = {
    "name": "geom",
    "atom_encoder": {
        "H": 0,
        "B": 1,
        "C": 2,
        "N": 3,
        "O": 4,
        "F": 5,
        "Al": 6,
        "Si": 7,
        "P": 8,
        "S": 9,
        "Cl": 10,
        "As": 11,
        "Br": 12,
        "I": 13,
        "Hg": 14,
        "Bi": 15,
    },
    "atom_decoder": ["H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi"],
    "max_n_nodes": 181,
    "with_h": True,
}

GEOM_WITHOUT_H = {
    "name": "geom",
    "atom_encoder": {
        "B": 0,
        "C": 1,
        "N": 2,
        "O": 3,
        "F": 4,
        "Al": 5,
        "Si": 6,
        "P": 7,
        "S": 8,
        "Cl": 9,
        "As": 10,
        "Br": 11,
        "I": 12,
        "Hg": 13,
        "Bi": 14,
    },
    "atom_decoder": ["B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi"],
    "max_n_nodes": 91,
    "with_h": False,
}


def canonical_eval_dataset_name(dataset_name):
    name = dataset_name.lower()
    if name == "qm9":
        return "qm9"
    if name in {"geom", "drug", "b3lyp", "b3lyp_17m", "b3lyp17m"}:
        return "geom"
    raise ValueError(f"Unsupported evaluation dataset: {dataset_name}")


def get_dataset_info(dataset_name, remove_h=False):
    name = canonical_eval_dataset_name(dataset_name)
    if name == "qm9":
        return QM9_WITHOUT_H if remove_h else QM9_WITH_H
    if name == "geom":
        return GEOM_WITHOUT_H if remove_h else GEOM_WITH_H
    raise ValueError(f"Unsupported evaluation dataset: {dataset_name}")
