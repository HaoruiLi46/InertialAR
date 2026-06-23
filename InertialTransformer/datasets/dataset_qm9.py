import os
import copy
from itertools import repeat
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Sequence
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem
from scipy.constants import physical_constants
from torch.utils.data import Dataset
import torch.nn.functional as F
import csv
from InertialTransformer.datasets.functional_group_extract_qm9 import analyze_smiles, HEADERS
from InertialTransformer.datasets.functional_group_mapping import build_functional_group_mapping, save_condition_mapping, load_condition_mapping
from InertialTransformer.datasets.tokenization_dictionary import VOCAB

from InertialTransformer.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D, mol_to_graph_data_obj_simple_3D_canonical
from InertialTransformer.utils import build_inertial_frame_and_rotate

class DatasetInertialSeqQM9(Dataset):
    def __init__( 
        self, root, dataset, task,
        position_pad_token=0, position_eos_token=0,
        max_seq_len=48,
        calculate_thermo=True):
        self.root = root
        self.transform = None
        self.raw_dir = os.path.join(root, "raw")
        self.processed_dir = os.path.join(self.root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.processed_file = os.path.join(self.processed_dir, "geometric_data_processed.pt")
        self.functional_group_counts_file = os.path.join(self.processed_dir, "functional_group_counts.csv")
        self.vocab = VOCAB
        self._indices = None
        self.remove_center = False

        self.target_field = [
            "mu",  "alpha", "homo", "lumo", "gap", "r2",
            "zpve", "u0", "u298", "h298", "g298", "cv",
        ]
        self.pd_target_field = [
            "mu", "alpha", "homo", "lumo", "gap", "r2",
            "zpve", "u0", "u298", "h298", "g298", "cv",
        ]
        self.task = task
        if self.task == "qm9":
            self.task_id = None
        else:
            self.task_id = self.target_field.index(task)
        self.calculate_thermo = calculate_thermo
        self.atom_dict = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

        self.hartree2eV = physical_constants["hartree-electron volt relationship"][0]

        self.conversion = {
            "mu": 1.0,
            "alpha": 1.0,
            "homo": self.hartree2eV,
            "lumo": self.hartree2eV,
            "gap": self.hartree2eV,
            "gap_02": self.hartree2eV,
            "r2": 1.0,
            "zpve": self.hartree2eV,
            "u0": self.hartree2eV,
            "u298": self.hartree2eV,
            "h298": self.hartree2eV,
            "g298": self.hartree2eV,
            "cv": 1.0,
        }

        super().__init__()

        self.preprocess()
        self.dataset = dataset
        data = torch.load(self.processed_file)
        self.x_list = data["x"]
        self.y_list = data["y"]
        self.positions_list = data["positions"]
        self.I_list = data["I"]

        functional_group_counts_list = pd.read_csv(self.functional_group_counts_file)
        self.functional_group_counts_list = functional_group_counts_list.to_numpy()

        self.max_seq_len = max_seq_len
        self.position_pad_token = position_pad_token
        self.position_eos_token = position_eos_token
        
        self.condition_mapping_file = os.path.join(self.processed_dir, "condition_ids.pt")
        if not os.path.exists(self.condition_mapping_file):
            self.condition_ids, self.pattern_to_id_dict, self.id_to_pattern_dict = build_functional_group_mapping(self.functional_group_counts_list[:, :-1])
            save_condition_mapping(self.condition_ids, self.pattern_to_id_dict, self.id_to_pattern_dict, self.processed_dir)
        else:
            self.condition_ids, self.pattern_to_id_dict, self.id_to_pattern_dict = load_condition_mapping(self.processed_dir)

        self.num_classes = len(self.pattern_to_id_dict) 

        return

    def mean(self):
        # y = torch.stack([self.y_list[i] for i in range(len(self))], dim=0)
        y = torch.stack([self.y_list[i] for i in self._indices], dim=0)
        y = y.mean(dim=0)
        return y

    def std(self):
        # y = torch.stack([self.y_list[i] for i in range(len(self))], dim=0)
        y = torch.stack([self.y_list[i] for i in self._indices], dim=0)
        y = y.std(dim=0)
        return y

    def get(self, idx):
        x = self.x_list[idx]
        positions = self.positions_list[idx]
        
        y = self.y_list[idx]
        condition_id = self.condition_ids[idx]
        condition_position = torch.full((1, 3), self.position_pad_token, dtype=positions.dtype)
        
        eos_position = torch.full((1, 3), self.position_eos_token, dtype=positions.dtype)
        padding_position = torch.full((1, 3), self.position_pad_token, dtype=positions.dtype)
        x = torch.cat([torch.tensor([self.vocab["<cls>"]], dtype=x.dtype), x, torch.tensor([self.vocab["<eos>"]], dtype=x.dtype)])
        positions = torch.cat([condition_position, positions, eos_position])
        
        L = x.shape[0]

        if L > self.max_seq_len:
            raise ValueError(
                f"Molecule at index {idx} has length {L-1} + 1 (EOS) = {L}, which exceeds max_seq_len {self.max_seq_len}. "
            )
        input_ids = x[:-1]
        input_positions = positions[:-1]
        targets = x[1:] 
        targets_positions = positions[1:]
        return input_ids, input_positions, targets, targets_positions, condition_id, y

    def indices(self):
        return range(len(self)) if self._indices is None else self._indices

    def index_select(self, idx):
        indices = self.indices()

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def __getitem__(self, idx):
        if (isinstance(idx, (int, np.integer)) or (isinstance(idx, torch.Tensor) and idx.dim() == 0) or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            data = self.get(self.indices()[idx])
            return data

        else:
            return self.index_select(idx)

    def __len__(self):
        if self._indices == None:
            return len(self.y_list)
        else:
            return len(self._indices)

    @property
    def raw_file_names(self):
        return [
            "gdb9.sdf",
            "gdb9.sdf.csv",
            "uncharacterized.txt",
            "qm9.csv",
            "atomref.txt",
        ]

    @property
    def raw_paths(self):
        files = self.raw_file_names
        return [os.path.join(self.raw_dir, f) for f in files]

    def get_thermo_dict(self):
        gdb9_txt_thermo = self.raw_paths[4]
        # Loop over file of thermochemical energies
        therm_targets = ["zpve", "u0", "u298", "h298", "g298", "cv"]
        therm_targets = [6, 7, 8, 9, 10, 11]

        # Dictionary that
        id2charge = self.atom_dict

        # Loop over file of thermochemical energies
        therm_energy = {target: {} for target in therm_targets}
        with open(gdb9_txt_thermo) as f:
            for line in f:
                # If line starts with an element, convert the rest to a list of energies.
                split = line.split()

                # Check charge corresponds to an atom
                if len(split) == 0 or split[0] not in id2charge.keys():
                    continue

                # Loop over learning targets with defined thermochemical energy
                for therm_target, split_therm in zip(therm_targets, split[1:]):
                    therm_energy[therm_target][id2charge[split[0]]] = float(split_therm)

        return therm_energy

    def preprocess(self):
        if os.path.exists(self.processed_file):
            return
    
        therm_energy = self.get_thermo_dict()
        print("therm_energy\t", therm_energy)

        df = pd.read_csv(self.raw_paths[1])
        df = df[self.pd_target_field]

        target = df.to_numpy()
        target = torch.tensor(target, dtype=torch.float)

        with open(self.raw_paths[2], "r") as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        data_df = pd.read_csv(self.raw_paths[3])
        whole_smiles_list = data_df["smiles"].tolist()
        print("TODO\t", whole_smiles_list[:100])

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        print("suppl: {}\tsmiles_list: {}".format(len(suppl), len(whole_smiles_list)))

        x_list, positions_list, I_list, y_list, functional_group_counts_list = [], [], [], [], []
        data_smiles_list, data_name_list, mol_idx, invalid_count = [], [], 0, 0
        for i, raw_mol in enumerate(tqdm(suppl)):
            if i in skip:
                invalid_count += 1
                continue
            if raw_mol is None:
                print(f"===== RDKit failed to parse molecule {i} (skip) =====")
                invalid_count += 1
                continue
            
            mol = Chem.Mol(raw_mol)
            smiles = Chem.MolToSmiles(mol)

            order = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
            reorder_mol = Chem.RenumberAtoms(mol,order)

            data, atom_count = mol_to_graph_data_obj_simple_3D(reorder_mol, pure_atomic_num=True)

            inertial_frame, rotated_positions = build_inertial_frame_and_rotate(data.positions)
            if inertial_frame is None:  # coplanar
                print(f"========== Coplanar system {i}, {data.positions.shape} (skip) ==========")
                continue

            data.id = torch.tensor([mol_idx])
            temp_y = target[i]
            if self.calculate_thermo:
                for atom, count in atom_count.items():
                    if atom not in self.atom_dict.values():
                        continue
                    for target_id, atom_sub_dic in therm_energy.items():
                        temp_y[target_id] -= atom_sub_dic[atom] * count

            # convert units
            for idx, col in enumerate(self.target_field):
                temp_y[idx] *= self.conversion[col]

            name = mol.GetProp("_Name")
            smiles = whole_smiles_list[i]

            temp_mol = AllChem.MolFromSmiles(smiles)
            if temp_mol is None:
                print("Exception with (invalid mol)\t", i)
                invalid_count += 1
                continue
            
            functional_group_counts = analyze_smiles(smiles)

            x_list.append(data.x)
            y_list.append(temp_y)
            I_list.append(inertial_frame)
            positions_list.append(rotated_positions)
            functional_group_counts_list.append(functional_group_counts)
            
            data_smiles_list.append(smiles)
            data_name_list.append(name)
            mol_idx += 1

        print(
            "mol id: [0, {}]\tlen of smiles: {}\tlen of set(smiles): {}".format(
                mol_idx - 1, len(data_smiles_list), len(set(data_smiles_list))
            )
        )
        print("{} invalid molecules".format(invalid_count))

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, "smiles.csv")
        print("saving to {}".format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data_name_series = pd.Series(data_name_list)
        saver_path = os.path.join(self.processed_dir, "name.csv")
        print("saving to {}".format(saver_path))
        data_name_series.to_csv(saver_path, index=False, header=False)

        functional_group_headers = HEADERS
        with open(self.functional_group_counts_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(functional_group_headers)
            for counts in functional_group_counts_list:
                writer.writerow(counts)

        torch.save({
            "x": x_list, "y": y_list, "positions": positions_list, "I": I_list
        }, self.processed_file)

        return
