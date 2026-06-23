import torch
from torch.utils.data import Dataset


class InertialARDataset(Dataset):
    """
    Wrap a processed molecular sequence dataset and pack variable-length samples
    into the flattened batch format expected by InertialAR.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collater(self, items):
        if not items:
            return {}

        sample_size = len(items[0])
        if sample_size == 6:
            (
                input_ids_list,
                positions_list,
                targets_list,
                targets_positions_list,
                condition_id_list,
                y_list,
            ) = zip(*items)
        elif sample_size == 5:
            (
                input_ids_list,
                positions_list,
                targets_list,
                targets_positions_list,
                condition_id_list,
            ) = zip(*items)
            y_list = None
        else:
            raise ValueError(f"Unsupported dataset sample size: {sample_size}")

        lens = [len(ids) for ids in input_ids_list]
        cu_seqlens = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
        max_len = max(lens)
        index_pos = torch.cat([torch.arange(length, dtype=torch.long) for length in lens])

        batch = {
            "input_ids": torch.cat(input_ids_list),
            "positions": torch.cat(positions_list),
            "targets": torch.cat(targets_list),
            "targets_positions": torch.cat(targets_positions_list),
            "condition_id": torch.tensor(condition_id_list, dtype=torch.long),
            "cu_seqlens": cu_seqlens,
            "index_pos": index_pos,
            "max_len": max_len,
        }

        if y_list is not None:
            batch["y"] = torch.stack(y_list)

        return batch
