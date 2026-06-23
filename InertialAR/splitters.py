import numpy as np
import torch


def _random_permutation(num_items, seed):
    np.random.seed(seed)
    print("using seed\t", seed)
    return np.random.permutation(num_items)


def _split_indices(num_items, seed, num_train=None, train_fraction=None, test_fraction=0.1):
    all_idx = _random_permutation(num_items, seed)

    if num_train is None:
        if train_fraction is None:
            raise ValueError("Either num_train or train_fraction must be provided.")
        num_train = int(train_fraction * num_items)

    num_test = int(test_fraction * num_items)
    num_valid = num_items - (num_train + num_test)
    if num_valid < 0:
        raise ValueError(
            f"Split requires at least {num_train} training items, got {num_items}."
        )

    train_idx = all_idx[:num_train]
    valid_idx = all_idx[num_train : num_train + num_valid]
    test_idx = all_idx[num_train + num_valid :]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_items

    return train_idx, valid_idx, test_idx


def split_dataset_by_indices(dataset, train_idx, valid_idx, test_idx, smiles_list=None):
    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if smiles_list is None:
        return train_dataset, valid_dataset, test_dataset

    train_smiles = [smiles_list.iloc[i, 0] for i in train_idx]
    valid_smiles = [smiles_list.iloc[i, 0] for i in valid_idx]
    test_smiles = [smiles_list.iloc[i, 0] for i in test_idx]
    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        (train_smiles, valid_smiles, test_smiles),
    )


def split_condition_ids_by_indices(
    condition_id_list, train_idx, valid_idx, test_idx, smiles_list=None
):
    cond_arr = np.asarray(condition_id_list)
    train_condition_ids = cond_arr[train_idx].tolist()
    valid_condition_ids = cond_arr[valid_idx].tolist()
    test_condition_ids = cond_arr[test_idx].tolist()

    if smiles_list is None:
        return train_condition_ids, valid_condition_ids, test_condition_ids

    train_smiles = [smiles_list.iloc[i, 0] for i in train_idx]
    valid_smiles = [smiles_list.iloc[i, 0] for i in valid_idx]
    test_smiles = [smiles_list.iloc[i, 0] for i in test_idx]
    return (
        train_condition_ids,
        valid_condition_ids,
        test_condition_ids,
        (train_smiles, valid_smiles, test_smiles),
    )


def split_qm9_indices(num_items, seed=0):
    return _split_indices(num_items, seed=seed, num_train=100000)


def split_b3lyp_indices(num_items, seed=0):
    return _split_indices(num_items, seed=seed, train_fraction=0.8)
