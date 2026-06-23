import json
import os
from collections import Counter

import torch
from tqdm import tqdm


def build_functional_group_mapping(functional_group_counts_list, verbose=False):
    """
    Build a mapping from functional-group count patterns to condition IDs.

    Returns:
        tuple: (condition_ids, pattern_to_id_dict, id_to_pattern_dict)
    """
    pattern_to_id_dict = {}
    id_to_pattern_dict = {}
    condition_ids = []
    next_available_id = 0

    if verbose:
        print(f"Processing {len(functional_group_counts_list)} molecules...")

    for mol_idx, functional_group_counts in tqdm(
        enumerate(functional_group_counts_list),
        total=len(functional_group_counts_list),
        desc="Processing molecules",
        disable=not verbose,
    ):
        for count in functional_group_counts:
            if isinstance(count, str):
                raise ValueError(f"Count is a string: {count}")

        binary_pattern = "".join(["1" if count > 0 else "0" for count in functional_group_counts])

        if binary_pattern in pattern_to_id_dict:
            condition_id = pattern_to_id_dict[binary_pattern]
        else:
            condition_id = next_available_id
            pattern_to_id_dict[binary_pattern] = condition_id
            id_to_pattern_dict[condition_id] = binary_pattern
            next_available_id += 1

            if verbose and next_available_id <= 10:
                print(f"New pattern #{condition_id}: '{binary_pattern}' (from molecule {mol_idx})")

        condition_ids.append(condition_id)

    if verbose:
        print("Mapping completed:")
        print(f"  - Total molecules processed: {len(functional_group_counts_list)}")
        print(f"  - Unique functional group patterns found: {len(pattern_to_id_dict)}")
        print(f"  - Condition embedding size needed: {len(pattern_to_id_dict)}")

        pattern_frequency = Counter(condition_ids)
        most_common_patterns = pattern_frequency.most_common(5)
        print("  - Most frequent patterns:")
        for cond_id, freq in most_common_patterns:
            pattern = id_to_pattern_dict[cond_id]
            print(f"    Condition ID {cond_id} (pattern: '{pattern}'): {freq} molecules")

    return condition_ids, pattern_to_id_dict, id_to_pattern_dict


def save_condition_mapping(condition_ids, pattern_to_id_dict, id_to_pattern_dict, save_dir, verbose=False):
    os.makedirs(save_dir, exist_ok=True)

    condition_ids_tensor = torch.tensor(condition_ids, dtype=torch.long)
    torch.save(condition_ids_tensor, os.path.join(save_dir, "condition_ids.pt"))

    with open(os.path.join(save_dir, "pattern_to_id.json"), "w") as f:
        json.dump(pattern_to_id_dict, f, indent=2)

    with open(os.path.join(save_dir, "id_to_pattern.json"), "w") as f:
        id_to_pattern_str_keys = {str(k): v for k, v in id_to_pattern_dict.items()}
        json.dump(id_to_pattern_str_keys, f, indent=2)

    if verbose:
        print(f"Condition mapping saved to {save_dir}")
        print(f"  - condition_ids.pt: {len(condition_ids)} condition IDs")
        print(f"  - pattern_to_id.json: {len(pattern_to_id_dict)} pattern->ID mappings")
        print(f"  - id_to_pattern.json: {len(id_to_pattern_dict)} ID->pattern mappings")


def load_condition_mapping(load_dir, verbose=False):
    condition_ids = torch.load(os.path.join(load_dir, "condition_ids.pt"))

    with open(os.path.join(load_dir, "pattern_to_id.json"), "r") as f:
        pattern_to_id_dict = json.load(f)

    with open(os.path.join(load_dir, "id_to_pattern.json"), "r") as f:
        id_to_pattern_str_keys = json.load(f)
        id_to_pattern_dict = {int(k): v for k, v in id_to_pattern_str_keys.items()}

    if verbose:
        print(f"Condition mapping loaded from {load_dir}")
        print(f"  - {len(condition_ids)} condition IDs loaded")
        print(f"  - {len(pattern_to_id_dict)} unique patterns")

    return condition_ids.tolist(), pattern_to_id_dict, id_to_pattern_dict


def analyze_condition_distribution(
    condition_ids,
    functional_group_name_list,
    id_to_pattern_dict=None,
    min_freq_filter=None,
    verbose=False,
):
    if not condition_ids:
        if verbose:
            print("Warning: condition_ids is empty.")
        return []

    original_total_count = len(condition_ids)
    frequency = Counter(condition_ids)

    if min_freq_filter is not None and isinstance(min_freq_filter, int):
        filtered_frequency = {
            condition_id: count
            for condition_id, count in frequency.items()
            if count >= min_freq_filter
        }
        filtered_total_count = sum(filtered_frequency.values())
        removed_count = original_total_count - filtered_total_count
    else:
        filtered_frequency = frequency
        filtered_total_count = original_total_count
        removed_count = 0

    sorted_conditions = sorted(filtered_frequency.items(), key=lambda item: item[1], reverse=True)

    if verbose:
        filter_str = f">= {min_freq_filter}" if min_freq_filter is not None else "none"
        print(f"\n--- Condition ID distribution (filter: {filter_str}) ---")
        print(
            f"Original molecules: {original_total_count} | "
            f"Filtered molecules: {filtered_total_count} | Removed: {removed_count}"
        )
        print(f"Unique patterns: {len(frequency)} -> {len(filtered_frequency)}")
        print(f"Conditions: {len(sorted_conditions)}")

        if len(sorted_conditions) <= 100:
            for rank, (condition_id, count) in enumerate(sorted_conditions, 1):
                percentage = (count / filtered_total_count) * 100 if filtered_total_count > 0 else 0

                print("-" * 80)
                print(f"Rank: {rank:<4} | ID: {condition_id:<5} | Count: {count:<8} ({percentage:>5.2f}%)")

                if id_to_pattern_dict and condition_id in id_to_pattern_dict:
                    binary_pattern = id_to_pattern_dict[condition_id]
                    print(f"Pattern: '{binary_pattern}'")

                    present_groups = [
                        name
                        for name, char in zip(functional_group_name_list, binary_pattern)
                        if char == "1"
                    ]

                    if present_groups:
                        print(f"Functional groups: {', '.join(present_groups)}")
                    else:
                        print("Functional groups: none")
                else:
                    print("Pattern: N/A")

        print("-" * 80)
    return sorted_conditions
