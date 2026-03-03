#!/usr/bin/env python3
"""
Analyze experiment results with multi-persona, multi-phase aggregation.

Computes per-phase, per-condition aggregate metrics:
  - Exact match rate (mean +/- std across personas)
  - Within-1 rate
  - Mean Absolute Error (MAE)
  - 95% confidence intervals

Handles both single-persona legacy results and multi-persona experiments.

Usage:
    python analyze_results.py results/experiment_YYYYMMDD_HHMMSS.json
    python analyze_results.py results/experiment_*.json  # multiple files
"""

import json
import math
import re
import sys
from collections import defaultdict


def extract_scale_value(text):
    """Extract numeric scale value from response text."""
    patterns = [
        r'\*\*(\d+)\*\*',
        r'\b(\d)\s*[—–-]',
        r'\ba?\s*\*?\*?(\d)\*?\*?\b',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1))
    return None


def smart_match(response_text, gt_label, gt_code, response_options):
    """Match response to ground truth. Returns (exact_match, within_1, extracted_code)."""
    text = response_text.lower().strip()
    gt_label_lower = gt_label.lower().strip()
    gt_code_int = int(gt_code) if gt_code.isdigit() else None

    n_options = len(response_options)
    is_scale = n_options >= 5

    extracted_code = None

    if is_scale and gt_code_int is not None:
        val = extract_scale_value(response_text)
        if val is not None:
            extracted_code = val
            exact = (val == gt_code_int)
            within1 = abs(val - gt_code_int) <= 1
            return exact, within1, extracted_code

    for code, label in sorted(response_options.items(), key=lambda x: -len(x[1])):
        if len(label) > 2 and label.lower() in text:
            extracted_code = int(code) if code.isdigit() else None
            exact = (code == gt_code) or (label.lower() == gt_label_lower)
            if gt_code_int and extracted_code:
                within1 = abs(extracted_code - gt_code_int) <= 1
            else:
                within1 = exact
            return exact, within1, extracted_code

    if gt_label_lower in text or (len(gt_label_lower) > 3 and gt_label_lower[:20] in text):
        return True, True, gt_code_int

    keyword_maps = {
        "some of the time": 4,
        "never": 5,
        "always": 1,
        "most of the time": 2,
        "about half": 3,
        "few big interests": 1,
        "benefit of all": 2,
        "waste a lot": 1,
        "waste some": 2,
        "don't waste": 3,
        "agree strongly": 1,
        "agree somewhat": 2,
        "neither agree nor disagree": 3,
        "disagree somewhat": 4,
        "disagree strongly": 5,
        "gotten better": 1,
        "stayed about the same": 2,
        "about the same": 2,
        "gotten worse": 5,
        "better off": 1,
        "worse off": 2,
        "more services": 2,
        "fewer services": 1,
        "favor a great deal": 1,
        "favor it a great deal": 1,
        "favor somewhat": 2,
        "favor it somewhat": 2,
        "neither favor nor oppose": 3,
        "oppose somewhat": 4,
        "oppose a great deal": 5,
        "government plan": 1,
        "government insurance": 1,
        "private insurance": 2,
        "increase a great deal": 1,
        "increase a moderate": 2,
        "increase a little": 3,
        "kept about the same": 4,
        "keep about the same": 4,
        "decrease a little": 5,
        "decrease a moderate": 6,
        "decrease a great deal": 7,
        "a great deal": 1,
        "a lot": 2,
        "a moderate amount": 3,
        "a little": 4,
        "none at all": 5,
        "favor regulation": 1,
        "favor it": 1,
        "oppose regulation": 2,
    }

    for phrase, code_val in sorted(keyword_maps.items(), key=lambda x: -len(x[0])):
        if phrase in text:
            extracted_code = code_val
            exact = (str(code_val) == gt_code)
            if gt_code_int:
                within1 = abs(code_val - gt_code_int) <= 1
            else:
                within1 = exact
            return exact, within1, extracted_code

    return False, False, extracted_code


def mean_ci(values):
    """Compute mean and 95% CI for a list of values."""
    n = len(values)
    if n == 0:
        return 0, 0, 0
    m = sum(values) / n
    if n == 1:
        return m, 0, 0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    ci = 1.96 * se
    return m, std, ci


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.json> [results_file2.json ...]")
        sys.exit(1)

    results_files = sys.argv[1:]

    all_responses = []
    metadata = {}
    for fpath in results_files:
        with open(fpath) as f:
            data = json.load(f)
        valid = [r for r in data.get("responses", []) if "error" not in r or "answer" in r]
        all_responses.extend(valid)
        if not metadata:
            metadata = {k: v for k, v in data.items() if k != "responses"}
        else:
            for k, v in data.get("phase_configs", {}).items():
                metadata.setdefault("phase_configs", {})[k] = v

    with open("eval_items.json") as f:
        eval_items = json.load(f)

    gt_lookup = {}
    q_lookup = {}
    for item in eval_items:
        var = item["variable"]
        q_lookup[var] = item
        for pid, gt in item.get("ground_truth", {}).items():
            gt_lookup[(var, pid)] = gt

    has_phases = any("phase" in r for r in all_responses)

    if has_phases:
        group_key = lambda r: (r.get("phase", 0), r.get("phase_name", "?"), r["condition"])
    else:
        group_key = lambda r: (0, "default", r["condition"])

    groups = defaultdict(list)
    for r in all_responses:
        if "error" in r and "answer" not in r:
            continue
        groups[group_key(r)].append(r)

    # Per-persona stats within each group
    # group -> persona_id -> {exact, within1, total, abs_err_sum, scale_count}
    group_persona_stats = defaultdict(lambda: defaultdict(lambda: {
        "exact": 0, "within1": 0, "total": 0,
        "abs_err_sum": 0, "scale_count": 0,
    }))

    for gkey, resps in groups.items():
        for r in resps:
            qid = r.get("question_id")
            pid = r.get("persona_id")
            gt = gt_lookup.get((qid, pid))
            if not gt:
                continue

            q_info = q_lookup.get(qid)
            if not q_info:
                continue

            gt_code = gt["code"]
            gt_label = gt["label"]
            answer = r.get("answer", "")
            exact, w1, ext_code = smart_match(answer, gt_label, gt_code, q_info["response_options"])

            stats = group_persona_stats[gkey][pid]
            stats["total"] += 1
            if exact:
                stats["exact"] += 1
            if w1:
                stats["within1"] += 1

            gt_code_int = int(gt_code) if gt_code.isdigit() else None
            if ext_code is not None and gt_code_int is not None:
                stats["abs_err_sum"] += abs(ext_code - gt_code_int)
                stats["scale_count"] += 1

    # Print header
    print("=" * 100)
    print("AGGREGATE RESULTS (multi-persona)")
    if metadata:
        model = metadata.get("model", "?")
        temp = metadata.get("temperature", "?")
        seed = metadata.get("seed", "?")
        n_per = metadata.get("n_personas_per_phase", "?")
        print(f"Model: {model}  |  Temperature: {temp}  |  Seed: {seed}  |  Personas/phase: {n_per}")
    print("=" * 100)
    print()

    sorted_groups = sorted(groups.keys())

    print(f"{'Phase':<25s} {'Condition':<20s} {'N_pers':>6s} "
          f"{'Exact':>14s} {'Within-1':>14s} {'MAE':>14s}")
    print("-" * 100)

    summary_table = []

    for gkey in sorted_groups:
        phase_num, phase_name, cond = gkey
        persona_stats = group_persona_stats[gkey]

        exact_rates = []
        within1_rates = []
        maes = []

        for pid, stats in persona_stats.items():
            if stats["total"] > 0:
                exact_rates.append(stats["exact"] / stats["total"])
                within1_rates.append(stats["within1"] / stats["total"])
            if stats["scale_count"] > 0:
                maes.append(stats["abs_err_sum"] / stats["scale_count"])

        n_personas = len(persona_stats)
        exact_m, exact_s, exact_ci = mean_ci(exact_rates)
        w1_m, w1_s, w1_ci = mean_ci(within1_rates)
        mae_m, mae_s, mae_ci = mean_ci(maes)

        exact_str = f"{exact_m:.1%} +/- {exact_ci:.1%}"
        w1_str = f"{w1_m:.1%} +/- {w1_ci:.1%}"
        mae_str = f"{mae_m:.2f} +/- {mae_ci:.2f}" if maes else "N/A"

        phase_label = f"{phase_num}: {phase_name}"
        print(f"{phase_label:<25s} {cond:<20s} {n_personas:>6d} "
              f"{exact_str:>14s} {w1_str:>14s} {mae_str:>14s}")

        summary_table.append({
            "phase": phase_num, "phase_name": phase_name,
            "condition": cond, "n_personas": n_personas,
            "exact_mean": exact_m, "exact_std": exact_s, "exact_ci": exact_ci,
            "within1_mean": w1_m, "within1_std": w1_s, "within1_ci": w1_ci,
            "mae_mean": mae_m, "mae_std": mae_s, "mae_ci": mae_ci,
        })

    # Markdown table for easy copy-paste
    print()
    print("=" * 100)
    print("MARKDOWN TABLE (for paper)")
    print("=" * 100)
    print()
    temp_str = metadata.get("temperature", "?")
    print(f"Temperature: {temp_str}. N={metadata.get('n_personas_per_phase', '?')} personas per phase.")
    print()
    print("| Phase | Modules | Condition | N | Exact Match | Within-1 | MAE |")
    print("|-------|---------|-----------|---|-------------|----------|-----|")
    for row in summary_table:
        phase_cfg = None
        for pcfg_num, pcfg in sorted(
            ((int(k), v) for k, v in metadata.get("phase_configs", {}).items()),
            key=lambda x: x[0],
        ):
            if pcfg_num == row["phase"]:
                phase_cfg = pcfg
                break
        n_modules = phase_cfg["n_modules"] if phase_cfg else "?"
        print(f"| {row['phase']}: {row['phase_name']} "
              f"| {n_modules} "
              f"| {row['condition']} "
              f"| {row['n_personas']} "
              f"| {row['exact_mean']:.1%} ({row['exact_ci']:.1%}) "
              f"| {row['within1_mean']:.1%} ({row['within1_ci']:.1%}) "
              f"| {row['mae_mean']:.2f} ({row['mae_ci']:.2f}) |")

    # Per-domain breakdown
    print()
    print("=" * 100)
    print("PER-DOMAIN BREAKDOWN")
    print("=" * 100)
    print()

    domain_group_stats = defaultdict(lambda: {"exact": 0, "total": 0})

    for gkey, resps in groups.items():
        for r in resps:
            qid = r.get("question_id")
            pid = r.get("persona_id")
            domain = r.get("question_domain", "?")
            gt = gt_lookup.get((qid, pid))
            if not gt:
                continue
            q_info = q_lookup.get(qid)
            if not q_info:
                continue
            exact, _, _ = smart_match(r.get("answer", ""), gt["label"], gt["code"],
                                      q_info["response_options"])
            key = (gkey, domain)
            domain_group_stats[key]["total"] += 1
            if exact:
                domain_group_stats[key]["exact"] += 1

    for (gkey, domain), stats in sorted(domain_group_stats.items()):
        phase_num, phase_name, cond = gkey
        total = stats["total"]
        exact = stats["exact"]
        pct = f"{100*exact/total:.0f}%" if total else "N/A"
        cond_short = "BASE" if cond == "baseline_static" else "FRMW" if cond == "full_framework" else cond[:4].upper()
        print(f"  Phase {phase_num} [{cond_short:4s}] {domain:20s}: {exact}/{total} ({pct})")

    # Framework retrieval patterns (if applicable)
    fw_responses = [r for r in all_responses if r.get("condition") == "full_framework"
                    and "modules_retrieved" in r]
    if fw_responses:
        print()
        print("=" * 100)
        print("FRAMEWORK MODULE RETRIEVAL PATTERNS")
        print("=" * 100)
        print()
        skill_counts = {}
        mod_counts = {}
        total_mods = 0
        for r in fw_responses:
            sk = r.get("skill_selected", "N/A")
            skill_counts[sk] = skill_counts.get(sk, 0) + 1
            mods = r.get("modules_retrieved", [])
            total_mods += len(mods)
            for m in mods:
                mod_counts[m] = mod_counts.get(m, 0) + 1

        print("Skill distribution:")
        for sk, cnt in sorted(skill_counts.items(), key=lambda x: -x[1]):
            print(f"  {sk:<25} {cnt:>3} ({100*cnt/len(fw_responses):.0f}%)")

        print()
        if fw_responses:
            print(f"Modules per question: mean={total_mods/len(fw_responses):.1f}")
        print()
        print("Module usage frequency:")
        for mod, cnt in sorted(mod_counts.items(), key=lambda x: -x[1]):
            print(f"  {mod:<25} {cnt:>3} ({100*cnt/len(fw_responses):.0f}%)")


if __name__ == "__main__":
    main()
