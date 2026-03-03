#!/usr/bin/env python3
"""
Significance analysis for S3 framework vs baseline across phases.

Within each phase, the same 10 personas are tested under both conditions
(baseline_static vs full_framework), so we use paired tests:
  - Paired t-test (parametric)
  - Wilcoxon signed-rank test (non-parametric, appropriate for N=10)
  - Cohen's d effect size

Usage:
    python significance_analysis.py
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


RESULT_FILES = [
    "results/phase0_phase1_n10_seed2024.json",
    "results/phase2_n10_seed2024.json",
    "results/phase3_n10_seed2024.json",
]
EVAL_FILE = "eval_items.json"


def extract_scale_value(text):
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
        "some of the time": 4, "never": 5, "always": 1,
        "most of the time": 2, "about half": 3,
        "few big interests": 1, "benefit of all": 2,
        "waste a lot": 1, "waste some": 2, "don't waste": 3,
        "agree strongly": 1, "agree somewhat": 2,
        "neither agree nor disagree": 3,
        "disagree somewhat": 4, "disagree strongly": 5,
        "gotten better": 1, "stayed about the same": 2,
        "about the same": 2, "gotten worse": 5,
        "better off": 1, "worse off": 2,
        "more services": 2, "fewer services": 1,
        "favor a great deal": 1, "favor it a great deal": 1,
        "favor somewhat": 2, "favor it somewhat": 2,
        "neither favor nor oppose": 3,
        "oppose somewhat": 4, "oppose a great deal": 5,
        "government plan": 1, "government insurance": 1,
        "private insurance": 2,
        "increase a great deal": 1, "increase a moderate": 2,
        "increase a little": 3, "kept about the same": 4,
        "keep about the same": 4, "decrease a little": 5,
        "decrease a moderate": 6, "decrease a great deal": 7,
        "a great deal": 1, "a lot": 2, "a moderate amount": 3,
        "a little": 4, "none at all": 5,
        "favor regulation": 1, "favor it": 1, "oppose regulation": 2,
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


def cohens_d_paired(x, y):
    diff = np.array(x) - np.array(y)
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0


def format_p(p):
    if p < 0.001:
        return "p < .001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


def main():
    with open(EVAL_FILE) as f:
        eval_items = json.load(f)

    gt_lookup = {}
    q_lookup = {}
    for item in eval_items:
        var = item["variable"]
        q_lookup[var] = item
        for pid, gt in item.get("ground_truth", {}).items():
            gt_lookup[(var, pid)] = gt

    all_responses = []
    all_phase_configs = {}
    all_persona_samples = {}
    for fpath in RESULT_FILES:
        with open(fpath) as f:
            data = json.load(f)
        valid = [r for r in data.get("responses", []) if "error" not in r or "answer" in r]
        all_responses.extend(valid)
        for k, v in data.get("phase_configs", {}).items():
            all_phase_configs[int(k)] = v
        for k, v in data.get("phase_persona_samples", {}).items():
            all_persona_samples[int(k)] = v

    # Build per-persona, per-phase, per-condition stats
    # key: (phase, condition, persona_id)
    persona_stats = defaultdict(lambda: {
        "exact": 0, "within1": 0, "total": 0,
        "abs_err_sum": 0, "scale_count": 0,
    })

    for r in all_responses:
        qid = r.get("question_id")
        pid = r.get("persona_id")
        phase = r.get("phase", 0)
        cond = r["condition"]
        gt = gt_lookup.get((qid, pid))
        if not gt:
            continue
        q_info = q_lookup.get(qid)
        if not q_info:
            continue

        answer = r.get("answer", "")
        exact, w1, ext_code = smart_match(answer, gt["label"], gt["code"],
                                          q_info["response_options"])
        key = (phase, cond, pid)
        persona_stats[key]["total"] += 1
        if exact:
            persona_stats[key]["exact"] += 1
        if w1:
            persona_stats[key]["within1"] += 1
        gt_code_int = int(gt["code"]) if gt["code"].isdigit() else None
        if ext_code is not None and gt_code_int is not None:
            persona_stats[key]["abs_err_sum"] += abs(ext_code - gt_code_int)
            persona_stats[key]["scale_count"] += 1

    phases = sorted(all_phase_configs.keys())

    print("=" * 90)
    print("SIGNIFICANCE ANALYSIS: S3 Framework vs Baseline")
    print("Paired tests (same personas under both conditions within each phase)")
    print("=" * 90)

    for phase in phases:
        cfg = all_phase_configs[phase]
        personas = all_persona_samples.get(phase, [])
        n = len(personas)

        print(f"\n{'─' * 90}")
        print(f"Phase {phase}: {cfg['name']} ({cfg['description']})")
        print(f"N = {n} personas, 22 questions each")
        print(f"{'─' * 90}")

        baseline_exact, framework_exact = [], []
        baseline_w1, framework_w1 = [], []
        baseline_mae, framework_mae = [], []

        for pid in personas:
            bkey = (phase, "baseline_static", pid)
            fkey = (phase, "full_framework", pid)
            bs = persona_stats[bkey]
            fs = persona_stats[fkey]

            if bs["total"] > 0:
                baseline_exact.append(bs["exact"] / bs["total"])
                baseline_w1.append(bs["within1"] / bs["total"])
            if fs["total"] > 0:
                framework_exact.append(fs["exact"] / fs["total"])
                framework_w1.append(fs["within1"] / fs["total"])
            if bs["scale_count"] > 0:
                baseline_mae.append(bs["abs_err_sum"] / bs["scale_count"])
            if fs["scale_count"] > 0:
                framework_mae.append(fs["abs_err_sum"] / fs["scale_count"])

        for metric_name, bl_vals, fw_vals in [
            ("Exact Match", baseline_exact, framework_exact),
            ("Within-1", baseline_w1, framework_w1),
            ("MAE", baseline_mae, framework_mae),
        ]:
            bl = np.array(bl_vals)
            fw = np.array(fw_vals)
            if len(bl) < 2 or len(fw) < 2 or len(bl) != len(fw):
                print(f"\n  {metric_name}: insufficient paired data (baseline={len(bl)}, framework={len(fw)})")
                continue

            bl_mean = bl.mean()
            fw_mean = fw.mean()
            diff = fw - bl
            diff_mean = diff.mean()

            # Paired t-test
            t_stat, t_p = stats.ttest_rel(fw, bl)

            # Wilcoxon signed-rank test
            try:
                w_stat, w_p = stats.wilcoxon(fw, bl, alternative='two-sided')
            except ValueError:
                w_stat, w_p = float('nan'), float('nan')

            d = cohens_d_paired(fw, bl)

            # Shapiro-Wilk normality test on differences
            if len(diff) >= 3:
                sw_stat, sw_p = stats.shapiro(diff)
            else:
                sw_stat, sw_p = float('nan'), float('nan')

            direction = "Framework > Baseline" if diff_mean > 0 else "Baseline > Framework"
            if metric_name == "MAE":
                direction = "Framework < Baseline (better)" if diff_mean < 0 else "Baseline < Framework (better)"

            print(f"\n  {metric_name}:")
            print(f"    Baseline mean:  {bl_mean:.3f} (SD = {bl.std(ddof=1):.3f})")
            print(f"    Framework mean: {fw_mean:.3f} (SD = {fw.std(ddof=1):.3f})")
            print(f"    Difference:     {diff_mean:+.3f}  [{direction}]")
            print(f"    Paired t-test:  t({len(bl)-1}) = {t_stat:.3f}, {format_p(t_p)}")
            print(f"    Wilcoxon:       W = {w_stat:.1f}, {format_p(w_p)}")
            print(f"    Cohen's d:      {d:.3f}")
            print(f"    Normality (Shapiro-Wilk on diffs): W = {sw_stat:.3f}, {format_p(sw_p)}")

            sig_marker = ""
            recommended_test = "Wilcoxon" if sw_p < 0.05 else "Paired t-test"
            if metric_name == "MAE":
                relevant_p = w_p if recommended_test == "Wilcoxon" else t_p
            else:
                relevant_p = w_p if recommended_test == "Wilcoxon" else t_p

            if relevant_p < 0.05:
                sig_marker = " *"
            if relevant_p < 0.01:
                sig_marker = " **"
            if relevant_p < 0.001:
                sig_marker = " ***"

            print(f"    >> Recommended test: {recommended_test}, {format_p(relevant_p)}{sig_marker}")

    # Summary table
    print(f"\n{'=' * 90}")
    print("SUMMARY TABLE")
    print(f"{'=' * 90}")
    print(f"{'Phase':<30s} {'Metric':<15s} {'BL Mean':>8s} {'FW Mean':>8s} {'Diff':>8s} "
          f"{'t-test p':>10s} {'Wilcoxon p':>10s} {'d':>7s}")
    print("-" * 90)

    for phase in phases:
        cfg = all_phase_configs[phase]
        personas = all_persona_samples.get(phase, [])

        baseline_exact, framework_exact = [], []
        baseline_w1, framework_w1 = [], []
        baseline_mae, framework_mae = [], []

        for pid in personas:
            bkey = (phase, "baseline_static", pid)
            fkey = (phase, "full_framework", pid)
            bs = persona_stats[bkey]
            fs = persona_stats[fkey]
            if bs["total"] > 0:
                baseline_exact.append(bs["exact"] / bs["total"])
                baseline_w1.append(bs["within1"] / bs["total"])
            if fs["total"] > 0:
                framework_exact.append(fs["exact"] / fs["total"])
                framework_w1.append(fs["within1"] / fs["total"])
            if bs["scale_count"] > 0:
                baseline_mae.append(bs["abs_err_sum"] / bs["scale_count"])
            if fs["scale_count"] > 0:
                framework_mae.append(fs["abs_err_sum"] / fs["scale_count"])

        for metric_name, bl_vals, fw_vals in [
            ("Exact", baseline_exact, framework_exact),
            ("Within-1", baseline_w1, framework_w1),
            ("MAE", baseline_mae, framework_mae),
        ]:
            bl = np.array(bl_vals)
            fw = np.array(fw_vals)
            if len(bl) != len(fw) or len(bl) < 2:
                continue
            _, t_p = stats.ttest_rel(fw, bl)
            try:
                _, w_p = stats.wilcoxon(fw, bl, alternative='two-sided')
            except ValueError:
                w_p = float('nan')
            d = cohens_d_paired(fw, bl)
            diff = (fw - bl).mean()

            def p_str(p):
                if p < 0.001: return "<.001***"
                elif p < 0.01: return f"{p:.3f}** "
                elif p < 0.05: return f"{p:.3f}*  "
                else: return f"{p:.3f}   "

            label = f"{phase}: {cfg['name']}"
            print(f"{label:<30s} {metric_name:<15s} {bl.mean():>8.3f} {fw.mean():>8.3f} "
                  f"{diff:>+8.3f} {p_str(t_p):>10s} {p_str(w_p):>10s} {d:>+7.3f}")

    print()
    print("Note: * p < .05, ** p < .01, *** p < .001 (two-tailed)")
    print("      With N=10, Wilcoxon signed-rank is recommended over paired t-test.")
    print("      Cohen's d: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large")


if __name__ == "__main__":
    main()
