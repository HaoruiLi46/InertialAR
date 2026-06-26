"""No-novelty EDM-style evaluation for generated InertialAR molecules."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from InertialTransformer.datasets.tokenization_dictionary import ATOM_TYPE_MAPPINGS
from InertialAR.evaluation.eval_follow_edm.analyze import analyze_stability_for_molecules
from InertialAR.evaluation.eval_follow_edm.datasets_config import canonical_eval_dataset_name, get_dataset_info


def _token_mapping_for_dataset(dataset_name: str):
    names = [dataset_name, dataset_name.lower(), canonical_eval_dataset_name(dataset_name)]
    for name in names:
        mapping = ATOM_TYPE_MAPPINGS.get(name)
        if mapping is not None:
            return mapping
    raise ValueError(f"No atom-token mapping for dataset '{dataset_name}'")


def get_atom_symbol_dict(dataset_name: str, remove_h: bool):
    eval_name = canonical_eval_dataset_name(dataset_name)
    if eval_name == "qm9":
        if remove_h:
            return {"C": 0, "C@": 0, "C@@": 0, "N": 1, "O": 2, "F": 3}
        return {"H": 0, "C": 1, "C@": 1, "C@@": 1, "N": 2, "O": 3, "F": 4}
    if eval_name == "geom":
        if remove_h:
            return {
                "B": 0,
                "C": 1,
                "C@": 1,
                "C@@": 1,
                "N": 2,
                "O": 3,
                "F": 4,
                "Al": 5,
                "Si": 6,
                "Si@": 6,
                "Si@@": 6,
                "P": 7,
                "P@": 7,
                "P@@": 7,
                "S": 8,
                "S@": 8,
                "S@@": 8,
                "Cl": 9,
                "As": 10,
                "Br": 11,
                "I": 12,
                "Hg": 13,
                "Bi": 14,
            }
        return {
            "H": 0,
            "B": 1,
            "C": 2,
            "C@": 2,
            "C@@": 2,
            "N": 3,
            "O": 4,
            "F": 5,
            "Al": 6,
            "Si": 7,
            "Si@": 7,
            "Si@@": 7,
            "P": 8,
            "P@": 8,
            "P@@": 8,
            "S": 9,
            "S@": 9,
            "S@@": 9,
            "Cl": 10,
            "As": 11,
            "Br": 12,
            "I": 13,
            "Hg": 14,
            "Bi": 15,
        }
    raise ValueError(f"Unsupported evaluation dataset: {dataset_name}")


def invariant_seq_for_edm_eval(dataset_name="qm9", input_path="seq.txt", remove_h=False, symbols_beyond_type=False):
    atom_dict = get_atom_symbol_dict(dataset_name, remove_h)

    with open(input_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    num_samples = len(lines)

    all_len_files = [len(line.split()) for line in lines]
    if len(all_len_files) == 0:
        print("Warning: empty input file", file=sys.stderr)
        return {
            "one_hot": torch.zeros((0, 1, 1)),
            "x": torch.zeros((0, 1, 3)),
            "node_mask": torch.zeros((0, 1, 1)),
        }

    max_num_atoms = math.ceil(max(all_len_files) / 4.0)
    print("max_num_atoms(max_len_seq/4):", max_num_atoms, file=sys.stderr)

    all_len = []
    num_type = max(atom_dict.values()) + 1
    count_invalid_len = 0
    count_invalid_seq = 0
    count_invalid_coords = 0
    one_hot = torch.zeros((num_samples, max_num_atoms, num_type), dtype=float)
    x = torch.zeros((num_samples, max_num_atoms, 3), dtype=float)
    node_mask = torch.zeros((num_samples, max_num_atoms, 1), dtype=float)
    idx = 0

    with open(input_path, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Parsing molecules"):
            if not symbols_beyond_type:
                mol = np.array(line.split())
                try:
                    mol = mol.reshape(-1, 4)
                except Exception:
                    for cut_idx in range(max(int(len(mol) / 4) - 1, 0)):
                        vals = mol[4 * cut_idx : 4 * cut_idx + 4]
                        try:
                            atom_dict[vals[0]]
                            vals[1:4].astype(float)
                        except Exception:
                            mol = mol[: 4 * cut_idx].reshape(-1, 4)
                            break
                        if cut_idx == int(len(mol) / 4) - 2:
                            mol = mol[: 4 * cut_idx + 4].reshape(-1, 4)
                    count_invalid_len += 1
                seq = mol[:, 0] if len(mol) > 0 else []
            else:
                try:
                    match = re.findall(r"\b[A-Za-z]+ [+-]?\d+.\d+ [+-]?\d+.\d+ [+-]?\d+.\d+\b", line)
                    mol = np.array([item.split() for item in match])
                    seq = mol[:, 0]
                except Exception:
                    count_invalid_seq += 1
                    continue

            try:
                one_hot_emb = torch.nn.functional.one_hot(torch.tensor([atom_dict[key] for key in seq]), num_type)
            except Exception:
                count_invalid_seq += 1
                continue
            try:
                invariant_coords = mol[:, 1:4].astype(float)
            except Exception:
                count_invalid_coords += 1
                continue

            num_nodes = len(seq)
            all_len.append(num_nodes)
            one_hot[idx, :num_nodes] = one_hot_emb
            x[idx, :num_nodes] = torch.tensor(invariant_coords)
            node_mask[idx, :num_nodes] = 1.0
            idx += 1

    one_hot, x, node_mask = one_hot[:idx], x[:idx], node_mask[:idx]
    print("max_num_atoms(after filter out invalid molecules):", 0 if len(all_len) == 0 else max(all_len), file=sys.stderr)
    print(
        "invalid: 1. length is not a multiple of 4; 2. invalid atom type; 3. invalid coords:",
        count_invalid_len,
        count_invalid_seq,
        count_invalid_coords,
        file=sys.stderr,
    )
    return {"one_hot": one_hot, "x": x, "node_mask": node_mask}


def convert_npz_to_text(npz_input_path: str, text_output_path: str, dataset_name: str) -> bool:
    if not os.path.exists(npz_input_path):
        print(f"Error: NPZ file not found: {npz_input_path}")
        return False

    atom_type_mapping = _token_mapping_for_dataset(dataset_name)
    print(f"Converting NPZ to text: {npz_input_path}")
    print(f"Using atom type mapping for '{dataset_name}': {atom_type_mapping}")

    try:
        data = np.load(npz_input_path, allow_pickle=True)
        if "molecules" in data:
            molecules_list = data["molecules"]
        elif "idx" in data and "pos" in data:
            molecules_list = [
                {"atom_types": idx_arr, "coordinates": pos_arr}
                for idx_arr, pos_arr in zip(data["idx"], data["pos"])
            ]
        else:
            print(f"Error: unsupported NPZ format. Keys: {list(data.keys())}")
            return False
    except Exception as exc:
        print(f"Error loading NPZ: {exc}")
        return False

    all_lines = []
    error_count = 0
    for mol_dict in molecules_list:
        try:
            atom_types_num = mol_dict["atom_types"]
            coordinates = mol_dict["coordinates"]
            if not isinstance(coordinates, np.ndarray) or coordinates.ndim != 2 or coordinates.shape[1] != 3:
                error_count += 1
                continue
            if len(atom_types_num) != len(coordinates):
                error_count += 1
                continue

            line_parts = []
            for at_num, coord_vec in zip(atom_types_num, coordinates):
                symbol = atom_type_mapping.get(int(at_num))
                if symbol is None:
                    raise ValueError(f"unknown atom token {at_num}")
                line_parts.append(symbol)
                line_parts.extend([str(float(c)) for c in coord_vec])
            all_lines.append(" ".join(line_parts))
        except Exception:
            error_count += 1
            continue

    output_dir = os.path.dirname(text_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(text_output_path, "w", encoding="utf-8") as handle:
        for line in all_lines:
            handle.write(line + "\n")

    print(f"Conversion complete: {len(all_lines)} molecules saved, {error_count} errors")
    print(f"Output: {text_output_path}")
    return True


def run_evaluation(
    input_path: str,
    dataset_name: str,
    rep_type: str = "invariant",
    remove_h: bool = False,
    symbols_beyond_type: bool = False,
    seed: int = 42,
):
    from datetime import datetime

    eval_dataset_name = canonical_eval_dataset_name(dataset_name)
    print("\n" + "=" * 60)
    print("Running EDM-style Generation Evaluation")
    print(f"  Input: {input_path}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Eval dataset: {eval_dataset_name}")
    print("=" * 60 + "\n")

    report_lines = [
        "=" * 70,
        "       MOLECULAR GENERATION EVALUATION REPORT",
        "=" * 70,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input file: {input_path}",
        f"Dataset: {dataset_name}",
        f"Eval dataset: {eval_dataset_name}",
        f"Rep type: {rep_type}",
        f"Remove H: {remove_h}",
        f"Seed: {seed}",
        "Novelty: not evaluated",
        "=" * 70,
        "",
    ]

    dataset_info = get_dataset_info(eval_dataset_name, remove_h=remove_h)
    if rep_type != "invariant":
        raise ValueError(f"Unsupported rep_type: {rep_type}")

    molecules = invariant_seq_for_edm_eval(
        dataset_name=eval_dataset_name,
        input_path=input_path,
        remove_h=remove_h,
        symbols_beyond_type=symbols_beyond_type,
    )

    num_molecules = molecules["one_hot"].shape[0]
    report_lines.extend([f"[Data] Valid molecules parsed: {num_molecules}", ""])

    stability_dict, rdkit_metrics = analyze_stability_for_molecules(molecules, dataset_info, None)

    print("\n" + "=" * 60)
    print("Stability Results:")
    print(stability_dict)

    results = {"stability": stability_dict}
    report_lines.extend(["-" * 70, "                    STABILITY METRICS", "-" * 70])
    for key, value in stability_dict.items():
        report_lines.append(f"  {key:<30}: {value:.6f}")
        print(f"  {key:<30}: {value:.6f}")
    report_lines.append("")

    dir_path, filename = os.path.split(input_path)
    file_base, _ = os.path.splitext(filename)

    if rdkit_metrics is not None:
        rdkit_values, unique_smiles, log_smiles = rdkit_metrics
        validity, uniqueness = rdkit_values
        results["validity"] = validity
        results["uniqueness"] = uniqueness
        results["unique_smiles"] = unique_smiles
        results["log_smiles"] = log_smiles

        print("\nRDKit Metrics:")
        print(f"  Validity:   {validity:.4f}")
        print(f"  Uniqueness: {uniqueness:.4f}")

        report_lines.extend(
            [
                "-" * 70,
                "                    RDKIT METRICS",
                "-" * 70,
                f"  {'Validity':<30}: {validity:.6f}",
                f"  {'Uniqueness':<30}: {uniqueness:.6f}",
                "",
                f"  {'Valid molecules':<30}: {int(validity * num_molecules)}",
                f"  {'Unique molecules':<30}: {len(unique_smiles)}",
                "",
                "-" * 70,
                "                    OUTPUT FILES",
                "-" * 70,
            ]
        )

        for name, smiles_list in [("unique", unique_smiles), ("log", log_smiles)]:
            save_path = os.path.join(dir_path, f"{file_base}_{name}_smiles.csv")
            with open(save_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for smiles in smiles_list:
                    writer.writerow([smiles])
            print(f"Saved {name} SMILES to: {save_path}")
            report_lines.append(f"  {name}_smiles.csv: {save_path}")
    else:
        print("\nWarning: install RDKit to obtain Validity and Uniqueness metrics")
        report_lines.append("[Warning] RDKit not available - skipping validity/uniqueness")

    report_path = os.path.join(dir_path, f"{file_base}_eval_report.txt")
    report_lines.extend(["", "=" * 70, "                    EVALUATION COMPLETE", "=" * 70])
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))
    print(f"\n[Report] Evaluation report saved to: {report_path}")
    results["report_path"] = report_path
    print("=" * 60 + "\n")
    return results


def write_summary_json(results, input_path: str, dataset_name: str, seed: int, summary_path: str | None = None):
    file_base, _ = os.path.splitext(input_path)
    if summary_path is None:
        summary_path = f"{file_base}_summary.json"

    stability = results.get("stability", {})
    metrics = {}
    for key in ("mol_stable", "atm_stable"):
        if key in stability:
            metrics[key] = float(stability[key])
    if "validity" in results:
        metrics["Validity"] = float(results["validity"])
    if "uniqueness" in results:
        metrics["Uniqueness"] = float(results["uniqueness"])

    summary = {
        "dataset_name": dataset_name,
        "eval_dataset": canonical_eval_dataset_name(dataset_name),
        "seed": int(seed),
        "input_path": input_path,
        "report_path": results.get("report_path"),
        "metrics": metrics,
    }

    output_dir = os.path.dirname(summary_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[Summary] Evaluation summary saved to: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="No-novelty EDM-style molecular generation evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="qm9", help="qm9, geom, drug, b3lyp, or b3lyp_17m")
    parser.add_argument("--rep_type", type=str, default="invariant", choices=["invariant"])
    parser.add_argument("--remove_h", default=False, action="store_true")
    parser.add_argument("--symbols_beyond_type", default=False, action="store_true")
    parser.add_argument("--input_path", type=str, default=None, help="Text file with atom x y z sequences")
    parser.add_argument("--input_npz", type=str, default=None, help="Generated NPZ file from InertialAR generation")
    parser.add_argument("--output_text", type=str, default=None, help="Text output path when --input_npz is provided")
    parser.add_argument("--summary_json", type=str, default=None, help="Summary JSON output path")
    args = parser.parse_args()

    if args.input_npz is None and args.input_path is None:
        parser.error("Provide either --input_npz or --input_path")

    input_path = args.input_path
    if args.input_npz is not None:
        if args.output_text is None:
            input_path = os.path.splitext(args.input_npz)[0] + ".txt"
        else:
            input_path = args.output_text
        if not convert_npz_to_text(args.input_npz, input_path, args.dataset_name):
            raise SystemExit(1)

    results = run_evaluation(
        input_path=input_path,
        dataset_name=args.dataset_name,
        rep_type=args.rep_type,
        remove_h=args.remove_h,
        symbols_beyond_type=args.symbols_beyond_type,
        seed=args.seed,
    )
    write_summary_json(results, input_path, args.dataset_name, args.seed, args.summary_json)
    return results


if __name__ == "__main__":
    main()
