#!/usr/bin/env python3
"""Simple extractor that prints only the differing parts of `i` across records.

This minimal script removes CLI parsing. To use, edit INPUT_FILE (and optional
LIMIT) below and run the script directly.

Behavior:
 - reads the NDJSON file line-by-line
 - skips records where key 'r' contains 1e+10
 - extracts numeric vectors from 'i' (one vector per numeric-containing sublist)
 - computes which indices of each vector position vary across the collected records
 - prints, per record and per vector, only the differing indices and their values

Edit the INPUT_FILE variable below to point at your JSON file.
"""

import json
import re
from typing import Any, List, Dict
import numpy as np

# --- configure here -------------------------------------------------
# change this filename directly to operate on a different file

# set LIMIT to an int to limit how many records to collect for comparison (None = all)
LIMIT = None
# ---------------------------------------------------------------------

# Regex to find ints/floats (including scientific notation) inside strings
NUM_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?")


def extract_numbers_from_string(s: str) -> List[float]:
    return [float(x) for x in NUM_RE.findall(s)]


def extract_numbers(obj: Any) -> List[float]:
    out: List[float] = []
    if obj is None:
        return out
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(extract_numbers_from_string(obj))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(extract_numbers(v))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(extract_numbers(v))
    return out


def extract_vectors_per_sublist(obj: Any) -> List[List[float]]:
    vectors: List[List[float]] = []

    def is_numeric_container(x: Any) -> bool:
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return True
        if isinstance(x, str) and NUM_RE.search(x):
            return True
        if isinstance(x, (list, tuple)):
            for e in x:
                if is_numeric_container(e):
                    return True
        if isinstance(x, dict):
            for e in x.values():
                if is_numeric_container(e):
                    return True
        return False

    def walk(node: Any):
        if isinstance(node, (list, tuple)):
            if is_numeric_container(node):
                vec = extract_numbers(node)
                if vec:
                    vectors.append(vec)
                return
            for child in node:
                walk(child)
        elif isinstance(node, dict):
            for child in node.values():
                walk(child)

    walk(obj)
    return vectors


def has_one_e10(obj: Any) -> bool:
    if obj is None:
        return False
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj) == 1e10
    if isinstance(obj, str):
        for tok in NUM_RE.findall(obj):
            try:
                if float(tok) == 1e10:
                    return True
            except Exception:
                continue
        return False
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if has_one_e10(v):
                return True
        return False
    if isinstance(obj, dict):
        for v in obj.values():
            if has_one_e10(v):
                return True
        return False
    return False


def collect_all_records(path: str, limit: int = None):
    records: List[List[List[float]]] = []
    cost_list = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "r" in obj and has_one_e10(obj["r"]):
                continue
            i_val = obj.get("i")
            if i_val is None:
                continue
            vecs = extract_vectors_per_sublist(i_val)
            cost_repeat = obj.get("r")[0]
            cost = sum(cost_repeat) / len(cost_repeat)
            cost_list.append(cost)
            records.append(vecs)
            seen += 1
            if limit is not None and seen >= limit:
                break
    
    return records, cost_list


def compute_differing_positions(all_records: List[List[List[float]]]):
    max_vec_count = 0
    for rec in all_records:
        if len(rec) > max_vec_count:
            max_vec_count = len(rec)

    differing: Dict[int, List[int]] = {}
    for vec_idx in range(max_vec_count):
        slot_vectors: List[List[float]] = [rec[vec_idx] for rec in all_records if len(rec) > vec_idx]
        if not slot_vectors:
            differing[vec_idx] = []
            continue
        max_len = max(len(v) for v in slot_vectors)
        diff_positions: List[int] = []
        for pos in range(max_len):
            vals = []
            for v in slot_vectors:
                if pos < len(v):
                    vals.append(v[pos])
            if not vals:
                continue
            first = vals[0]
            all_same = all((x == first) for x in vals)
            if not all_same:
                diff_positions.append(pos)
        differing[vec_idx] = diff_positions
    return differing


def process_only_diff(path: str, limit: int = None):
    all_records, cost_list = collect_all_records(path, limit)

    differing = compute_differing_positions(all_records)
    rec_idx_list = []
    vec_idx_list = []
    diff_idx_list = []
    diff_values_list = []
    for rec_idx, rec in enumerate(all_records):
        for vec_idx, vec in enumerate(rec):
            diff_pos = differing.get(vec_idx, [])
            if not diff_pos:
                continue
            values = [vec[p] for p in diff_pos if p < len(vec)]
            

            rec_idx_list.append(rec_idx)
            vec_idx_list.append(vec_idx)
            diff_idx_list.append(diff_pos)
            diff_values_list.append(values)
            
            # print(json.dumps(out))
    

    rec_idx_list = np.array(rec_idx_list)
    vec_idx_list = np.array(vec_idx_list)
    diff_idx_list = np.array(diff_idx_list)
    diff_values_list = np.array(diff_values_list)
    cost_list = np.array(cost_list)
    
    out = {
            "record_index": rec_idx_list,
            "vector_index": vec_idx_list,
            "diff_indices": diff_idx_list,
            "diff_values": diff_values_list,
            "cost": cost_list
        }
    return out
    


if __name__ == "__main__":
    # Run the simplified only-diff flow using the constants above.
    INPUT_FILE = "/root/work/tenset/dataset/measure_records/k80/([0c9a5ba46ffc5e1a9e5641018527117f,4,7,7,160,1,1,160,960,1,1,1,960,4,7,7,960],cuda).json"
    out = process_only_diff(INPUT_FILE, LIMIT)

    save_file = "i_vectors_diffs.npz"
    np.savez(save_file, **out)