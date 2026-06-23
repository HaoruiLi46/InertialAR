import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Sequence
import copy
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import csv
from InertialTransformer.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D, mol_to_graph_data_obj_simple_3D_canonical
from InertialTransformer.utils import build_inertial_frame_and_rotate
from InertialTransformer.datasets.tokenization_dictionary import VOCAB
from InertialTransformer.datasets.functional_group_extract_b3lyp import analyze_smiles_file
from InertialTransformer.datasets.functional_group_mapping import build_functional_group_mapping, save_condition_mapping, load_condition_mapping



class DatasetInertialSeqB3LYP(Dataset):
    def __init__(
        self, root, max_seq_len=-1, max_coordinate=-1,
        position_pad_token=0, position_eos_token=0, target_size=1000000):
        self.root = root
        self.max_seq_len = max_seq_len
        self.max_coordinate = max_coordinate
        self.position_pad_token = position_pad_token
        self.position_eos_token = position_eos_token
        self.vocab = VOCAB
        self.target_size = target_size
        self.raw_dir = os.path.join(root, "raw")
        self.processed_dir = os.path.join(self.root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.processed_file = os.path.join(self.processed_dir, "geometric_data_processed.pt")
        self.smiles_file = os.path.join(self.processed_dir, "smiles.pt")
        self.functional_group_counts_file = os.path.join(self.processed_dir, "functional_group_counts.npy")
        self.functional_group_header_csv_file = os.path.join(self.processed_dir, "functional_group_header.csv")

        self._indices = None
        self.remove_center = False

        super().__init__()

        self.preprocess()

        self.condition_mapping_file = os.path.join(self.processed_dir, "condition_ids.pt")

        if not os.path.exists(self.condition_mapping_file):
            self.functional_group_counts_list, self.functional_group_name_list = self.functional_group_processing()
            self.condition_ids, self.pattern_to_id_dict, self.id_to_pattern_dict = build_functional_group_mapping(self.functional_group_counts_list)
            save_condition_mapping(self.condition_ids, self.pattern_to_id_dict, self.id_to_pattern_dict, self.processed_dir)
        else:
            with open(self.functional_group_header_csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                self.functional_group_name_list = next(reader)
            self.condition_ids, self.pattern_to_id_dict, self.id_to_pattern_dict = load_condition_mapping(self.processed_dir)
        self.num_classes = len(self.pattern_to_id_dict) 

        print("loading data...")
        data = torch.load(self.processed_file)
        self.x_list = data["x"]
        self.positions_list = data["positions"]
        self.I_list = data["I"]

        print("Finished as per expected.")
        return


    def preprocess(self):
        if os.path.exists(self.processed_file):
            print("Processed file already exists.")
            return

        raw_file_list = sorted(os.listdir(self.raw_dir))
        raw_file_list = [x for x in raw_file_list if x.endswith(".json")]

        if not raw_file_list:
            print("No raw files found.")
            return

        print(f"Found {len(raw_file_list)} raw files. Processing to create {self.target_size} samples...")

        x_list, positions_list, I_list, smiles_list = [], [], [], []

        num_fail_create_mol = 0
        num_fail_create_inertial_frame = 0
        num_fail_equal_smiles = 0
        num_fail_create_mol_from_smiles = 0

        sample_count = 0
        samples_needed = self.target_size

        for raw_file in tqdm(raw_file_list):
            if sample_count >= samples_needed:
                break

            raw_file_path = os.path.join(self.raw_dir, raw_file)
            try:
                with open(raw_file_path, 'r') as f:
                    data_list = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not read or parse {raw_file_path}, skipping. Error: {e}")
                continue

            for data in data_list:
                if sample_count >= samples_needed:
                    break

                if not all(k in data for k in ["coordinates", "atomic-numbers"]):
                    print(f"========== Missing coordinates or atomic-numbers, (skip) ==========")
                    continue
                positions = data["coordinates"]
                x = np.array(data["atomic-numbers"]) - 1

                if len(x) * 3 != len(positions) or len(x) == 0:
                    print(f"========== Invalid coordinates or atomic-numbers, (skip) ==========")
                    continue

                mol = self.create_canonical_mol_from_json_data(data)
                if mol is None:
                    num_fail_create_mol += 1
                    continue

                mol_data, atom_count = mol_to_graph_data_obj_simple_3D(mol, pure_atomic_num=True)
                
                inertial_frame, rotated_positions = build_inertial_frame_and_rotate(mol_data.positions)
                if inertial_frame is None:  # coplanar
                    num_fail_create_inertial_frame += 1
                    continue
                mol_from_json_no_H = Chem.RemoveHs(mol)
                smiles_from_json_standardized = Chem.MolToSmiles(mol_from_json_no_H, canonical=True)

                pubchem_smiles = data["pubchem-obabel-canonical-smiles"]

                try:
                    mol_from_pubchem = Chem.MolFromSmiles(pubchem_smiles)
                    if mol_from_pubchem is None:
                        num_fail_create_mol_from_smiles += 1
                        continue
                    smiles_from_pubchem_standardized = Chem.MolToSmiles(mol_from_pubchem, canonical=True)
                except Exception as e:
                    print(f"========== Failed to generate SMILES from PubChem mol: {e}, (skip) ==========")
                    continue

                if smiles_from_json_standardized != smiles_from_pubchem_standardized:
                    num_fail_equal_smiles += 1
                    continue

                x_list.append(mol_data.x)
                I_list.append(inertial_frame)
                positions_list.append(rotated_positions)
                smiles_list.append(smiles_from_json_standardized)
                sample_count += 1

        print(f"Collected {sample_count} samples from main files.")
        print(f"Number of failed to create canonical mol: {num_fail_create_mol}")
        print(f"Number of failed to create inertial frame: {num_fail_create_inertial_frame}")
        print(f"Number of failed to equal smiles: {num_fail_equal_smiles}")
        print(f"Number of failed to create mol from smiles: {num_fail_create_mol_from_smiles}")

        if sample_count < samples_needed:
            remaining_samples = samples_needed - sample_count
            raise ValueError(
                f"Insufficient data: collected {sample_count} samples but need {samples_needed} samples. "
                f"Missing {remaining_samples} samples. Please check your raw data files."
            )

        print(f"Successfully collected {sample_count} samples.")

        torch.save({
            "x": x_list,
            "positions": positions_list,
            "I": I_list
        }, self.processed_file)
        
        torch.save({
            "smiles": smiles_list
        }, self.smiles_file)

        print("Preprocessing complete.")
        return

    def get(self, idx):
        x = self.x_list[idx]
        positions = self.positions_list[idx]

        condition_id = self.condition_ids[idx]
        
        eos_position = torch.full((1, 3), self.position_eos_token, dtype=positions.dtype)
        padding_position = torch.full((1, 3), self.position_pad_token, dtype=positions.dtype)
        x = torch.cat([torch.tensor([self.vocab["<cls>"]], dtype=x.dtype), x, torch.tensor([self.vocab["<eos>"]], dtype=x.dtype)])
        positions = torch.cat([padding_position, positions, eos_position])

        L = x.shape[0]
        if L > self.max_seq_len and self.max_seq_len != -1:
            raise ValueError(
                f"Molecule at index {idx} has length {L-1} + 1 (EOS) = {L}, which exceeds max_seq_len {self.max_seq_len}. "
            )
        input_ids = x[:-1]
        input_positions = positions[:-1]
        targets = x[1:]
        targets_positions = positions[1:]

        y = None
        return input_ids, input_positions, targets, targets_positions, condition_id

    def functional_group_processing(self):
        if os.path.exists(self.functional_group_counts_file):
            print("Functional group counts file already exists.")
            functional_group_counts_list = np.load(self.functional_group_counts_file)
            with open(self.functional_group_header_csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                name_list_loaded = next(reader)
            return functional_group_counts_list, name_list_loaded
        
        name_list_loaded, functional_group_counts_list = analyze_smiles_file(self.smiles_file, self.functional_group_counts_file, self.functional_group_header_csv_file)
        return functional_group_counts_list, name_list_loaded

    def create_canonical_mol_from_json_data(self, data):
        """
        Create an RDKit molecule from JSON data containing atoms, bonds, and coordinates.
        """
        try:
            mol = Chem.RWMol()

            atomic_nums = data['atomic-numbers']
            for atomic_num in atomic_nums:
                mol.AddAtom(Chem.Atom(int(atomic_num)))

            connections = data.get('connection-indices', [])
            bond_orders = data.get('bond-order', [])

            bond_type_map = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
                4: Chem.BondType.AROMATIC
            }
            
            num_bonds = len(bond_orders)
            if len(connections) != num_bonds * 2:
                raise ValueError(
                    f"Bond index count ({len(connections)}) does not match bond order count ({num_bonds})."
                )

            for i in range(num_bonds):
                atom1_idx = connections[i * 2] - 1
                atom2_idx = connections[i * 2 + 1] - 1
                bond_order = bond_orders[i]
                bond_type = bond_type_map.get(bond_order, Chem.BondType.UNSPECIFIED)
                mol.AddBond(atom1_idx, atom2_idx, bond_type)

            mol = mol.GetMol()

            coords = np.array(data['coordinates']).reshape(-1, 3)
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for i, pos in enumerate(coords):
                conformer.SetAtomPosition(i, pos)
            mol.AddConformer(conformer, assignId=True)
            
            Chem.SanitizeMol(mol)
            
            smiles = Chem.MolToSmiles(mol, canonical=True)
            order = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
            reorder_mol = Chem.RenumberAtoms(mol,order)
            return reorder_mol
        except Exception as e:
            print(f"Failed to create molecule: {e}")
            return None

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
            return len(self.x_list)
        else:
            return len(self._indices)
