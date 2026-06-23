import msgpack
import os
import numpy as np
import torch
import json
import pickle

import copy
from itertools import repeat
import pandas as pd
from tqdm import tqdm
from typing import Sequence
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.constants import physical_constants
from torch.utils.data import Dataset
import torch.nn.functional as F
import csv
from InertialTransformer.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D
from InertialTransformer.utils import build_inertial_frame_and_rotate
from InertialTransformer.datasets.tokenization_dictionary import VOCAB
from InertialTransformer.datasets.functional_group_extract_drug import analyze_smiles_file
from InertialTransformer.datasets.functional_group_mapping import build_functional_group_mapping, save_condition_mapping, load_condition_mapping

class DatasetInertialSeqDrug(Dataset):
    def __init__(
        self, root, position_pad_token=0, position_eos_token=0,
        max_seq_len=64, val_proportion=0.1, test_proportion=0.1,
        filter_molecule_size=None, num_conformations=30):
        self.root = root
        self.transform = None
        self.filter_molecule_size = filter_molecule_size
        self.vocab = VOCAB
        self.position_pad_token = position_pad_token
        self.position_eos_token = position_eos_token
        self.max_seq_len = max_seq_len
        self.num_conformations = num_conformations
        self.raw_dir = os.path.join(root, "raw")
        self.processed_dir = os.path.join(self.root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.processed_file = os.path.join(self.processed_dir, "geometric_data_processed.pt")
        self.mask_file = os.path.join(self.processed_dir, "mask_list.pt")
        self.index_split_file = os.path.join(self.processed_dir, "index_split.pt")
        self.other_info_file = os.path.join(self.processed_dir, "other_info.pt")
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
        print("splitting data...")
        self.train_idx, self.val_idx, self.test_idx = self.split_data(val_proportion=val_proportion, test_proportion=test_proportion)
        
        self.x_list = data["x"]
        self.positions_list = data["positions"]
        self.I_list = data["I"]

        print("Finished as per expected.")
        return
    
    def split_data(self, val_proportion=0.1, test_proportion=0.1):
        """
        Build train/val/test indices over the compact processed dataset.
        """
        if not os.path.exists(self.processed_file):
            raise FileNotFoundError(f"Processed file {self.processed_file} not found. Please run preprocess() first.")

        if os.path.exists(self.index_split_file):
            print("Index split file already exists.")
            index_split = torch.load(self.index_split_file)
            return index_split["train_idx"], index_split["val_idx"], index_split["test_idx"]

        print("Calculating train/val/test indices...")
        
        perm_file = self.raw_paths[2]
        perm = np.load(perm_file).astype(int)
        mask_list = torch.load(self.mask_file)

        valid_perm = [idx for idx in perm if not mask_list[idx]]

        num_valid_mol = len(valid_perm)
        val_index = int(num_valid_mol * val_proportion)
        test_index = val_index + int(num_valid_mol * test_proportion)
        
        train_original_idx = valid_perm[test_index:]
        val_original_idx = valid_perm[:val_index]
        test_original_idx = valid_perm[val_index:test_index]

        valid_original_indices = np.where(np.array(mask_list) == False)[0]
        original_to_valid_map = {original_idx: valid_idx for valid_idx, original_idx in enumerate(valid_original_indices)}

        train_idx = [original_to_valid_map[i] for i in train_original_idx]
        val_idx = [original_to_valid_map[i] for i in val_original_idx]
        test_idx = [original_to_valid_map[i] for i in test_original_idx]

        torch.save({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}, self.index_split_file)

        return train_idx, val_idx, test_idx

    def get(self, idx):
        condition_id = self.condition_ids[idx]
        x = self.x_list[idx]
        positions = self.positions_list[idx]
        
        eos_position = torch.full((1, 3), self.position_eos_token, dtype=positions.dtype)
        pad_position = torch.full((1, 3), self.position_pad_token, dtype=positions.dtype)
        
        x = torch.cat([torch.tensor([self.vocab["<cls>"]], dtype=x.dtype), x, torch.tensor([self.vocab["<eos>"]], dtype=x.dtype)])
        positions = torch.cat([pad_position, positions, eos_position])
        
        L = x.shape[0]

        if L > self.max_seq_len:
            raise ValueError(
                f"Molecule at index {idx} has length {L-1} + 1 (EOS) = {L}, which exceeds max_seq_len {self.max_seq_len}. "
            )
        input_ids = x[:-1]
        input_positions = positions[:-1]
        targets = x[1:] 
        targets_positions = positions[1:]
        return input_ids, input_positions, targets, targets_positions, condition_id

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

    @property
    def raw_file_names(self):
        return [
            "drugs_crude.msgpack",
            "rdkit_folder/summary_drugs.json",
            "geom_permutation.npy",
            "rdkit_folder"
        ]

    @property
    def raw_paths(self):
        files = self.raw_file_names
        return [os.path.join(self.raw_dir, f) for f in files]

    def preprocess(self):
        if os.path.exists(self.processed_file):
            print("Processed file already exists.")
            return

        x_list = []
        positions_list = []
        I_list = []
        mask_list = []
        
        smiles_list = []
        mol_len_list = []
        x_ori_list = []
        positions_ori_list = []
        
        raw_file = self.raw_paths[0]
        unpacker = msgpack.Unpacker(open(raw_file, "rb"))
        drugs_file = self.raw_paths[1]
        with open(drugs_file, "r") as f:
            drugs_summ = json.load(f)
        count_error = 0
        count_num_conformer_error = 0
        count_conformer_error = 0
        count_inertial_frame_error = 0
        for i, drugs_1k in enumerate(unpacker):
            print('no file error, num_conf error, conf_xyz error, inertial_frame error:', count_error, count_num_conformer_error, count_conformer_error, count_inertial_frame_error)
            print(f"Unpacking file {i}...")
            for smiles, all_info in tqdm(drugs_1k.items()):
                try:
                    pickle_path = os.path.join(self.raw_paths[3], drugs_summ[smiles]['pickle_path'])
                    with open(pickle_path, 'rb') as f:
                        rdkit_data = pickle.load(f) 
                except:
                    count_error += 1
                    rdkit_data = None

                conformers = all_info['conformers']
                flag = False
                if rdkit_data != None:
                    rdkit_conformers = rdkit_data['conformers']
                    try:
                        assert len(conformers) == len(rdkit_conformers)
                    except:
                        count_num_conformer_error += 1
                        flag = True

                all_energies = []
                for conformer in conformers:
                    all_energies.append(conformer['totalenergy'])
                all_energies = np.array(all_energies)
                argsort = np.argsort(all_energies)
                lowest_energies = argsort[:self.num_conformations]
                for id in lowest_energies:
                    conformer = conformers[id]
                    coords = np.array(conformer['xyz']).astype(float)       # conformer['xyz']: atom type + xyz
                    if rdkit_data != None and flag != True:
                        mol = rdkit_conformers[id]['rd_mol']
                        rdkit_xyz = mol.GetConformer().GetPositions()
                        try:
                            assert abs(coords[:,1:] - rdkit_xyz).sum() < 0.1
                            Chem.MolToSmiles(mol)
                            order = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
                            reorder_mol = Chem.RenumberAtoms(mol,order)
                            atom_type = np.array([atom.GetSymbol() for atom in reorder_mol.GetAtoms()])
                            atomic_number = np.array([atom.GetAtomicNum() for atom in reorder_mol.GetAtoms()])
                            smiles_order_coords = reorder_mol.GetConformer().GetPositions()
                            smiles_order_coords = torch.tensor(smiles_order_coords, dtype=torch.float32)
                            inertial_frame, rotated_positions = build_inertial_frame_and_rotate(smiles_order_coords)
                            try:
                                assert inertial_frame is not None and rotated_positions is not None
                                x_list.append(torch.tensor(atomic_number - 1, dtype=torch.long)) ## Here H is 0. C is 5
                                positions_list.append(rotated_positions if isinstance(rotated_positions, torch.Tensor) else torch.tensor(rotated_positions, dtype=torch.float32))
                                I_list.append(inertial_frame if isinstance(inertial_frame, torch.Tensor) else torch.tensor(inertial_frame, dtype=torch.float32))
                                
                                smiles_list.append(smiles)
                                mol_len_list.append(len(coords))
                                x_ori_list.append(torch.tensor(coords[:,0] - 1, dtype=torch.long)) ## Here H is 0. C is 5
                                positions_ori_list.append(torch.tensor(coords[:,1:], dtype=torch.float32))
                                
                                mask_list.append(False)
                            except:
                                print(f"========== skip ==========")
                                count_inertial_frame_error += 1
                                flag = True
                                mask_list.append(True)
                                
                        
                        except:
                            count_conformer_error += 1
                            flag = True
                            mask_list.append(True)

                    else:
                        mask_list.append(True)
                    

        torch.save({
            "x": x_list, "positions": positions_list, "I": I_list}, self.processed_file)        
        
        torch.save({"smiles": smiles_list}, self.smiles_file)
        
        torch.save({
            "mol_len": mol_len_list,
            "x_ori": x_ori_list, "positions_ori": positions_ori_list}, self.other_info_file)
        
        torch.save(mask_list, self.mask_file)
        print('no file error, num_conf error, conf_xyz error, inertial_frame error:', count_error, count_num_conformer_error, count_conformer_error, count_inertial_frame_error)

        return

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
