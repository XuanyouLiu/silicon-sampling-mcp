"""
Experiment runner for the cognitive-grounded silicon sampling framework.
Uses Claude Agent SDK with Max subscription (no API key needed).
Full framework condition uses real MCP server for selective retrieval.

Experimental design (per advisor recommendations):
  - For each phase, randomly sample N personas from the pool
  - Within each phase, baseline and framework use the same persona sample
  - Different phases use different persona samples
  - Temperature is set and reported (default 1.0, per Argyle et al. 2023)

Four ablation conditions:
  (a) baseline_static: Full backstory in one block (Argyle et al. style)
  (b) rules_only: Identity anchor + constraints + full backstory
  (c) rules_skills: Rules + MCP Skill selection + full backstory provided
  (d) full_framework: Rules + MCP Skill + MCP selective module retrieval

Phase configurations (varying persona data richness):
  Phase 0 (Sparse):          7 modules, ~31 fields
  Phase 1 (Enriched+limits): 11 modules, ~107 fields
  Phase 2 (Enriched+free):   11 modules, ~107 fields (no retrieval limits)
  Phase 3 (Full):            13 modules, ~170 fields

Usage:
    python run_experiment.py --phases 2 3 --conditions baseline_static full_framework
    python run_experiment.py --phases 3 --n-personas 10 --temperature 1.0
    python run_experiment.py --phases 0 1 2 3 --conditions all --seed 42
"""

import json
import re
import sys
import random
import argparse
from pathlib import Path
from datetime import datetime

import anyio

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

DEMO_DIR = Path(__file__).resolve().parent
PERSONAS_DIR = DEMO_DIR / "personas"
RULES_DIR = DEMO_DIR / "rules"
SKILLS_DIR = DEMO_DIR / "skills"
PYTHON_BIN = str(DEMO_DIR / ".venv-linux" / "bin" / "python")
SERVER_PY = str(DEMO_DIR / "server.py")

VALID_MODULES = [
    "demographics", "life_narrative", "politics", "economy",
    "health", "social_context", "racial_attitudes",
    "values_personality", "media_consumption",
    "policy_positions", "civic_participation",
    "religion_community", "local_context",
]
ALL_CONDITIONS = ["baseline_static", "rules_only", "rules_skills", "full_framework"]

# ---------------------------------------------------------------------------
# Phase configurations: which modules are available per phase
# ---------------------------------------------------------------------------

PHASE_CONFIGS = {
    0: {
        "name": "Sparse",
        "modules": [
            "demographics", "life_narrative", "politics", "economy",
            "health", "social_context", "racial_attitudes",
        ],
        "description": "7 modules, ~31 fields per persona",
    },
    1: {
        "name": "Enriched + limits",
        "modules": [
            "demographics", "life_narrative", "politics", "economy",
            "health", "social_context", "racial_attitudes",
            "values_personality", "media_consumption",
            "religion_community", "local_context",
        ],
        "description": "11 modules, ~107 fields per persona",
    },
    2: {
        "name": "Enriched + free",
        "modules": [
            "demographics", "life_narrative", "politics", "economy",
            "health", "social_context", "racial_attitudes",
            "values_personality", "media_consumption",
            "religion_community", "local_context",
        ],
        "description": "11 modules, ~107 fields, free retrieval (no module count limits)",
    },
    3: {
        "name": "Full",
        "modules": VALID_MODULES,
        "description": "13 modules, ~170 fields per persona",
    },
}


def parse_reasoning_answer(text):
    """Split response into reasoning and answer parts."""
    m = re.search(r'ANSWER\s*:\s*(.+)', text, re.DOTALL)
    if m:
        answer = m.group(1).strip()
        reasoning_match = re.search(r'REASONING\s*:\s*(.+?)(?=ANSWER\s*:)', text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        return reasoning, answer
    return "", text.strip()


def load_survey_questions():
    eval_path = DEMO_DIR / "eval_items.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        return [
            {
                "id": item["variable"],
                "text": item["question_text"],
                "label": item["label"],
                "response_options": item.get("response_options", {}),
                "ground_truth": item.get("ground_truth", {}),
                "domain": item.get("domain", ""),
                "expected_skill": item.get("expected_skill", ""),
            }
            for item in items
        ]
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_personas(persona_ids=None):
    out = {}
    if not PERSONAS_DIR.exists():
        return out
    for fp in sorted(PERSONAS_DIR.glob("*.json")):
        pid = fp.stem
        if persona_ids and pid not in persona_ids:
            continue
        with open(fp, "r", encoding="utf-8") as f:
            out[pid] = json.load(f)
    return out


def filter_persona_for_phase(persona, phase_modules):
    """Return a copy of persona containing only the specified modules."""
    return {mod: persona[mod] for mod in phase_modules if mod in persona}


def load_text(directory, name):
    p = directory / f"{name}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


def render_rule(template, persona, module_filter=None):
    """Render a rule template with persona data, optionally filtering modules."""
    demo = persona.get("demographics", {})
    name = demo.get("name", "")
    if not name:
        name = f"a {demo.get('age', '?')}-year-old respondent"
    replacements = {
        "{name}": str(name),
        "{age}": str(demo.get("age", "?")),
        "{gender}": str(demo.get("gender", "person")),
        "{city}": str(demo.get("city", demo.get("state", "?"))),
        "{state}": str(demo.get("state", "?")),
        "{region}": str(demo.get("region", "?")),
        "{education}": str(demo.get("education", "?")),
        "{race}": str(demo.get("race", "?")),
        "{religion}": str(demo.get("religion", "?")),
    }
    if "{full_backstory}" in template:
        modules_to_render = module_filter if module_filter else VALID_MODULES
        parts = []
        for mod in modules_to_render:
            data = persona.get(mod)
            if not data:
                continue
            parts.append(f"[{mod}]")
            if isinstance(data, dict):
                for k, v in data.items():
                    parts.append(f"  {k}: {v}")
            else:
                parts.append(f"  {data}")
        replacements["{full_backstory}"] = "\n".join(parts)
    result = template
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result


def sample_personas_for_phases(all_persona_ids, n_per_phase, phases, seed):
    """Sample n personas per phase with different samples across phases.

    Within each phase, the same personas are used for all conditions.
    Across phases, different personas are sampled (when pool is large enough).

    Always allocates based on ALL_PHASES (0-3) so that persona assignment
    is stable regardless of which subset of phases is actually run.
    """
    all_phases = sorted(PHASE_CONFIGS.keys())
    rng = random.Random(seed)
    pool = sorted(all_persona_ids)

    if n_per_phase * len(all_phases) <= len(pool):
        rng.shuffle(pool)
        full_samples = {}
        offset = 0
        for phase in all_phases:
            full_samples[phase] = pool[offset:offset + n_per_phase]
            offset += n_per_phase
        return {p: full_samples[p] for p in phases}

    samples = {}
    for phase in phases:
        phase_rng = random.Random(seed + phase * 1000)
        shuffled = pool[:]
        phase_rng.shuffle(shuffled)
        samples[phase] = shuffled[:n_per_phase]
    return samples


def get_mcp_server_config(phase_modules=None):
    """Build MCP server config, optionally restricting modules for a phase."""
    args = [SERVER_PY]
    if phase_modules:
        args.extend(["--allowed-modules"] + phase_modules)
    return {"command": PYTHON_BIN, "args": args}


# ---------------------------------------------------------------------------
# Query runners for each condition
# ---------------------------------------------------------------------------


async def run_no_tools(system_prompt, question_text, model):
    """Conditions (a) baseline_static and (b) rules_only: no MCP tools."""
    opts = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=1,
        model=model,
        allowed_tools=[],
    )
    answer = ""
    usage_info = {}
    async for msg in query(prompt=question_text, options=opts):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    answer = block.text.strip()
        elif isinstance(msg, ResultMessage):
            usage_info = {
                "input_tokens": (msg.usage or {}).get("input_tokens", 0),
                "output_tokens": (msg.usage or {}).get("output_tokens", 0),
                "cache_creation_input_tokens": (msg.usage or {}).get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": (msg.usage or {}).get("cache_read_input_tokens", 0),
                "total_cost_usd": msg.total_cost_usd,
                "duration_ms": msg.duration_ms,
                "duration_api_ms": msg.duration_api_ms,
            }
    return answer, {}, usage_info


async def run_with_mcp(system_prompt, question_text, model,
                       allowed_tools, mcp_config, max_turns=5):
    """Conditions (c) rules_skills and (d) full_framework: real MCP tool calls."""
    opts = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=max_turns,
        model=model,
        mcp_servers={"survey": mcp_config},
        allowed_tools=allowed_tools,
    )
    answer = ""
    tool_log = []
    usage_info = {}

    async for msg in query(prompt=question_text, options=opts):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    answer = block.text.strip()
                elif isinstance(block, ToolUseBlock):
                    tool_log.append({
                        "tool": block.name,
                        "input": block.input if hasattr(block, "input") else {},
                    })
        elif isinstance(msg, ResultMessage):
            usage_info = {
                "input_tokens": (msg.usage or {}).get("input_tokens", 0),
                "output_tokens": (msg.usage or {}).get("output_tokens", 0),
                "cache_creation_input_tokens": (msg.usage or {}).get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": (msg.usage or {}).get("cache_read_input_tokens", 0),
                "total_cost_usd": msg.total_cost_usd,
                "duration_ms": msg.duration_ms,
                "duration_api_ms": msg.duration_api_ms,
            }

    return answer, tool_log, usage_info


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------


async def run_condition(condition, persona_id, persona, question, model,
                        phase_config):
    phase_modules = phase_config["modules"]

    result = {
        "condition": condition,
        "persona_id": persona_id,
        "question_id": question["id"],
        "question_domain": question.get("domain", ""),
        "expected_skill": question.get("expected_skill", ""),
        "model": model,
    }

    demo = persona.get("demographics", {})
    identity = (
        f"You are {persona_id}, a {demo.get('age','?')}-year-old "
        f"{demo.get('gender','person')} ({demo.get('race','?')}) "
        f"living in {demo.get('state','?')} ({demo.get('region','?')}). "
        f"Education: {demo.get('education','?')}. Religion: {demo.get('religion','?')}."
    )

    if condition == "baseline_static":
        template = load_text(RULES_DIR, "baseline_static")
        if not template:
            result["error"] = "baseline_static rule not found"
            return result
        system = render_rule(template, persona, module_filter=phase_modules)
        raw_answer, _, usage_info = await run_no_tools(system, question["text"], model)
        reasoning, answer = parse_reasoning_answer(raw_answer)
        result["raw_response"] = raw_answer
        result["reasoning"] = reasoning
        result["answer"] = answer
        result["usage"] = usage_info

    elif condition == "rules_only":
        template = load_text(RULES_DIR, "rules_only")
        if not template:
            result["error"] = "rules_only rule not found"
            return result
        system = render_rule(template, persona, module_filter=phase_modules)
        raw_answer, _, usage_info = await run_no_tools(system, question["text"], model)
        reasoning, answer = parse_reasoning_answer(raw_answer)
        result["raw_response"] = raw_answer
        result["reasoning"] = reasoning
        result["answer"] = answer
        result["usage"] = usage_info

    elif condition == "rules_skills":
        template = load_text(RULES_DIR, "rules_only")
        if not template:
            result["error"] = "rules_only rule not found"
            return result
        system = render_rule(template, persona, module_filter=phase_modules)
        system += (
            "\n\nBefore answering, use the get_survey_skill tool to select "
            "a reasoning approach. Then answer using that skill's guidance "
            "together with the background information above."
        )
        mcp_config = get_mcp_server_config(phase_modules)
        raw_answer, tool_log, usage_info = await run_with_mcp(
            system, question["text"], model,
            allowed_tools=["mcp__survey__get_survey_skill"],
            mcp_config=mcp_config,
            max_turns=3,
        )
        reasoning, answer = parse_reasoning_answer(raw_answer)
        result["raw_response"] = raw_answer
        result["reasoning"] = reasoning
        result["answer"] = answer
        result["usage"] = usage_info
        for entry in tool_log:
            if "get_survey_skill" in entry["tool"]:
                result["skill_selected"] = entry["input"].get("skill_type", "?")
                result["skill_reasoning"] = entry["input"].get("question_context", "")

    elif condition == "full_framework":
        template = load_text(RULES_DIR, "survey_respondent")
        if not template:
            result["error"] = "survey_respondent rule not found"
            return result
        system = template.replace("{persona_id}", persona_id)
        mcp_config = get_mcp_server_config(phase_modules)
        raw_answer, tool_log, usage_info = await run_with_mcp(
            system, question["text"], model,
            allowed_tools=[
                "mcp__survey__get_survey_skill",
                "mcp__survey__get_persona_modules",
            ],
            mcp_config=mcp_config,
            max_turns=5,
        )
        reasoning, answer = parse_reasoning_answer(raw_answer)
        result["raw_response"] = raw_answer
        result["reasoning"] = reasoning
        result["answer"] = answer
        result["usage"] = usage_info
        for entry in tool_log:
            if "get_survey_skill" in entry["tool"]:
                result["skill_selected"] = entry["input"].get("skill_type", "?")
                result["skill_reasoning"] = entry["input"].get("question_context", "")
            elif "get_persona_modules" in entry["tool"]:
                result["modules_retrieved"] = entry["input"].get("modules", [])
                result["retrieval_reasoning"] = entry["input"].get("question_context", "")

    else:
        result["error"] = f"Unknown condition: {condition}"

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_result(r):
    cond = r["condition"]
    answer = r.get("answer", r.get("error", "?"))
    reasoning = r.get("reasoning", "")
    usage = r.get("usage", {})
    parts = [f"  [{cond:16s}]"]
    if "skill_selected" in r:
        parts.append(f"skill={r['skill_selected']:22s}")
    if "modules_retrieved" in r:
        parts.append(f"modules={r['modules_retrieved']}")
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    cost = usage.get("total_cost_usd")
    parts.append(f"[in={in_tok} out={out_tok}" + (f" ${cost:.4f}]" if cost else "]"))
    print(" ".join(parts))
    if reasoning:
        print(f"{'':20s} R: {reasoning.replace(chr(10), ' ')[:200]}")
    print(f"{'':20s} A: {answer.replace(chr(10), ' ')[:200]}")


def print_summary(responses, questions):
    print("\n--- SUMMARY ---")

    by_cond = {}
    for r in responses:
        c = r["condition"]
        by_cond.setdefault(c, []).append(r)

    gt_map = {}
    for q in questions:
        if q.get("ground_truth"):
            gt_map[q["id"]] = q["ground_truth"]

    for cond, resps in by_cond.items():
        skills = {}
        module_counts = []
        matches = 0
        total_gt = 0
        for r in resps:
            if "skill_selected" in r:
                s = r["skill_selected"]
                skills[s] = skills.get(s, 0) + 1
            if "modules_retrieved" in r:
                module_counts.append(len(r["modules_retrieved"]))
            qid = r.get("question_id")
            pid = r.get("persona_id")
            if qid in gt_map and pid in gt_map[qid]:
                total_gt += 1
                gt_label = gt_map[qid][pid].get("label", "")
                ans = r.get("answer", "")
                if gt_label.lower() in ans.lower():
                    matches += 1

        total_in = sum(r.get("usage", {}).get("input_tokens", 0) for r in resps)
        total_out = sum(r.get("usage", {}).get("output_tokens", 0) for r in resps)
        total_cost = sum(r.get("usage", {}).get("total_cost_usd", 0) or 0 for r in resps)

        print(f"\n  [{cond}]")
        print(f"    Responses: {len(resps)}")
        if total_gt > 0:
            print(f"    Ground-truth matches: {matches}/{total_gt} ({100*matches/total_gt:.0f}%)")
        if skills:
            print(f"    Skills: {json.dumps(skills)}")
        if module_counts:
            avg = sum(module_counts) / len(module_counts)
            print(f"    Modules/question: mean={avg:.1f}, range={min(module_counts)}-{max(module_counts)}")
        print(f"    Tokens: input={total_in:,} output={total_out:,} total={total_in+total_out:,}")
        print(f"    Cost: ${total_cost:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(args):
    conditions = ALL_CONDITIONS if "all" in args.conditions else args.conditions
    phases = args.phases

    all_personas = load_personas(args.personas)
    if not all_personas:
        print("No personas found. Add JSON files to personas/.")
        return 1

    survey_questions = load_survey_questions()
    if not survey_questions:
        print("No eval items found. Run scripts/generate_personas.py first.")
        return 1

    all_persona_ids = sorted(all_personas.keys())
    n_available = len(all_persona_ids)
    n_per_phase = args.n_personas

    if n_per_phase > n_available:
        print(f"Warning: requested {n_per_phase} personas per phase "
              f"but only {n_available} available. Using all {n_available}.")
        n_per_phase = n_available

    phase_samples = sample_personas_for_phases(
        all_persona_ids, n_per_phase, phases, args.seed
    )

    results_dir = DEMO_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {
        "run_id": run_id,
        "model": args.model,
        "temperature": "SDK default (claude-agent-sdk does not expose temperature control)",
        "conditions": conditions,
        "phases": phases,
        "n_personas_per_phase": n_per_phase,
        "seed": args.seed,
        "total_personas_available": n_available,
        "phase_persona_samples": {str(p): phase_samples[p] for p in phases},
        "phase_configs": {
            str(p): {"name": PHASE_CONFIGS[p]["name"],
                      "n_modules": len(PHASE_CONFIGS[p]["modules"]),
                      "description": PHASE_CONFIGS[p]["description"]}
            for p in phases
        },
        "questions": [q["id"] for q in survey_questions],
        "n_questions": len(survey_questions),
        "repeats": args.repeats,
        "responses": [],
    }

    total = sum(len(phase_samples[p]) for p in phases) * len(survey_questions) * len(conditions) * args.repeats
    print(f"Experiment configuration:")
    print(f"  Model: {args.model} (temperature={args.temperature})")
    print(f"  Phases: {phases}")
    print(f"  Conditions: {conditions}")
    print(f"  Personas per phase: {n_per_phase}")
    print(f"  Questions: {len(survey_questions)}")
    print(f"  Repeats: {args.repeats}")
    print(f"  Total API calls: {total}")
    print(f"  Seed: {args.seed}")
    for p in phases:
        cfg = PHASE_CONFIGS[p]
        pids = phase_samples[p]
        print(f"  Phase {p} ({cfg['name']}): {len(cfg['modules'])} modules, "
              f"personas={pids}")
    print("=" * 70)

    done = 0
    for phase in phases:
        cfg = PHASE_CONFIGS[phase]
        phase_persona_ids = phase_samples[phase]
        phase_modules = cfg["modules"]

        print(f"\n{'='*70}")
        print(f"PHASE {phase}: {cfg['name']} ({cfg['description']})")
        print(f"Personas: {phase_persona_ids}")
        print(f"{'='*70}")

        for pid in phase_persona_ids:
            persona = all_personas[pid]
            demo = persona.get("demographics", {})
            label = f"{demo.get('age','?')}yo {demo.get('gender','?')}, {demo.get('race','?')}"
            print(f"\nPersona: {pid} ({label})")

            for q in survey_questions:
                gt = q.get("ground_truth", {}).get(pid)
                for _ in range(args.repeats):
                    for cond in conditions:
                        done += 1
                        try:
                            r = await run_condition(
                                cond, pid, persona, q, args.model, cfg,
                            )
                            r["phase"] = phase
                            r["phase_name"] = cfg["name"]
                            print_result(r)
                            all_results["responses"].append(r)
                        except Exception as e:
                            err = {
                                "condition": cond,
                                "persona_id": pid,
                                "question_id": q["id"],
                                "phase": phase,
                                "phase_name": cfg["name"],
                                "error": str(e),
                            }
                            print(f"  [{cond}] ERROR: {e}")
                            all_results["responses"].append(err)

                        if done % 20 == 0:
                            print(f"\n  --- Progress: {done}/{total} ---\n")

    out_path = results_dir / f"experiment_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Results saved to {out_path}")
    print(f"Total responses: {len(all_results['responses'])}")
    print_summary(all_results["responses"], survey_questions)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Silicon sampling experiment (Claude Agent SDK + Max subscription + MCP)"
    )
    parser.add_argument(
        "--model", default="sonnet",
        choices=["sonnet", "opus", "haiku"],
        help="Claude model (default: sonnet)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="LLM temperature (default: 1.0, per Argyle et al. 2023)",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=["baseline_static", "full_framework"],
        choices=ALL_CONDITIONS + ["all"],
    )
    parser.add_argument(
        "--phases", nargs="+", type=int, default=[3],
        choices=[0, 1, 2, 3],
        help="Which phases to run (default: 3). Each phase uses different persona data richness.",
    )
    parser.add_argument(
        "--n-personas", type=int, default=10,
        help="Number of personas to sample per phase (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=2024,
        help="Random seed for persona sampling (default: 2024)",
    )
    parser.add_argument("--personas", nargs="+", default=None,
                        help="Override: specific persona IDs to use (ignores sampling)")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    return anyio.run(async_main, args)


if __name__ == "__main__":
    sys.exit(main())
