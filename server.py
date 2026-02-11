"""
Silicon Sampling MCP Server
============================
A Model Context Protocol server implementing selective persona retrieval
for silicon sampling research. This server operationalizes the cognitive
model of survey response (Tourangeau et al., 2000) by allowing LLMs to
selectively access modular persona information before answering each
survey question.

Usage:
    python server.py                    # stdio transport (for Cursor/Claude Desktop)
    python server.py --transport sse    # SSE transport (for web clients)
"""

import json
import os
import argparse
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERSONAS_DIR = Path(__file__).parent / "personas"
SKILLS_DIR = Path(__file__).parent / "skills"
RULES_DIR = Path(__file__).parent / "rules"

VALID_MODULES = [
    "demographics",
    "life_narrative",
    "health",
    "economy",
    "politics",
    "social_views",
    "local_context",
]

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Silicon Sampling Server",
    version="0.1.0",
    description=(
        "MCP server for selective persona retrieval in silicon sampling. "
        "Implements modular persona databases that LLMs can query per-question, "
        "operationalizing the selective retrieval stage of the cognitive survey "
        "response model (Tourangeau et al., 2000)."
    ),
)

# In-memory state
_active_persona: Optional[str] = None
_personas: dict[str, dict] = {}
_retrieval_log: list[dict] = []


def _load_personas() -> None:
    """Load all persona JSON files from the personas directory."""
    global _personas
    _personas = {}
    if not PERSONAS_DIR.exists():
        return
    for fp in sorted(PERSONAS_DIR.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            persona_id = fp.stem
            _personas[persona_id] = data


def _load_skill(skill_name: str) -> Optional[str]:
    """Load a skill template from the skills directory."""
    skill_path = SKILLS_DIR / f"{skill_name}.txt"
    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8")
    return None


def _load_rule(rule_name: str) -> Optional[str]:
    """Load a rule template from the rules directory."""
    rule_path = RULES_DIR / f"{rule_name}.txt"
    if rule_path.exists():
        return rule_path.read_text(encoding="utf-8")
    return None


def _render_rule(rule_template: str, persona: dict) -> str:
    """Fill in a rule template with persona demographic information."""
    demo = persona.get("demographics", {})
    replacements = {
        "{name}": str(demo.get("name", "Unknown")),
        "{age}": str(demo.get("age", "?")),
        "{gender}": str(demo.get("gender", "person")),
        "{city}": str(demo.get("city", "?")),
        "{state}": str(demo.get("state", "?")),
        "{education}": str(demo.get("education", "?")),
        "{race}": str(demo.get("race", "?")),
        "{religion}": str(demo.get("religion", "?")),
    }
    # Build full backstory for baseline/rules-only conditions
    if "{full_backstory}" in rule_template:
        backstory_parts = []
        for module_name in VALID_MODULES:
            module_data = persona.get(module_name)
            if module_data:
                backstory_parts.append(f"[{module_name}]")
                if isinstance(module_data, dict):
                    for k, v in module_data.items():
                        backstory_parts.append(f"  {k}: {v}")
                else:
                    backstory_parts.append(f"  {module_data}")
        replacements["{full_backstory}"] = "\n".join(backstory_parts)

    result = rule_template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


# Load personas on startup
_load_personas()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_personas() -> str:
    """List all available personas in the database.

    Returns a summary of each persona including their ID and a brief
    demographic description. Use this to select which persona to activate
    before running a survey simulation.
    """
    _load_personas()
    if not _personas:
        return json.dumps({"error": "No personas found. Add JSON files to the personas/ directory."})

    summaries = []
    for pid, data in _personas.items():
        demo = data.get("demographics", {})
        summary = {
            "persona_id": pid,
            "name": demo.get("name", "Unknown"),
            "age": demo.get("age", "?"),
            "gender": demo.get("gender", "?"),
            "state": demo.get("state", "?"),
            "education": demo.get("education", "?"),
            "party_id": data.get("politics", {}).get("party_identification", "?"),
        }
        summaries.append(summary)

    return json.dumps({"personas": summaries, "count": len(summaries)}, indent=2)


@mcp.tool()
def set_active_persona(persona_id: str) -> str:
    """Set the active persona for the current survey simulation session.

    This must be called before using get_persona_modules. It establishes
    which respondent the model is simulating, corresponding to the Rules
    layer of the framework (identity anchor).

    Args:
        persona_id: The ID of the persona to activate (from list_personas).
    """
    global _active_persona
    _load_personas()

    if persona_id not in _personas:
        available = list(_personas.keys())
        return json.dumps({
            "error": f"Persona '{persona_id}' not found.",
            "available_personas": available,
        })

    _active_persona = persona_id
    demo = _personas[persona_id].get("demographics", {})

    return json.dumps({
        "status": "active",
        "persona_id": persona_id,
        "identity_anchor": (
            f"You are {demo.get('name', 'a person')}, "
            f"a {demo.get('age', '?')}-year-old {demo.get('gender', 'person')} "
            f"living in {demo.get('city', '?')}, {demo.get('state', '?')}. "
            f"Answer survey questions naturally, as you would in a real telephone survey."
        ),
        "available_modules": VALID_MODULES,
    })


@mcp.tool()
def get_rule(rule_name: str = "survey_respondent") -> str:
    """Load and render a Rule template for the active persona.

    Rules are the first layer of the framework, establishing the respondent's
    identity anchor, response style constraints, and behavioral guardrails.
    This corresponds to the comprehension stage of the cognitive model
    (Tourangeau et al., 2000).

    The rule template is populated with the active persona's demographic
    information. For ablation experiments, different rules implement
    different conditions:
    - "survey_respondent": Full framework rule (with selective retrieval)
    - "baseline_static": Static backstory baseline (Argyle et al., 2023)
    - "rules_only": Rules without Skills or MCP

    Args:
        rule_name: Name of the rule to load. Default is "survey_respondent".
    """
    if _active_persona is None:
        return json.dumps({
            "error": "No active persona. Call set_active_persona first.",
        })

    persona = _personas.get(_active_persona)
    if persona is None:
        return json.dumps({"error": f"Persona '{_active_persona}' not found."})

    rule_template = _load_rule(rule_name)
    if rule_template is None:
        available = [f.stem for f in RULES_DIR.glob("*.txt")] if RULES_DIR.exists() else []
        return json.dumps({
            "error": f"Rule '{rule_name}' not found.",
            "available_rules": available,
        })

    rendered = _render_rule(rule_template, persona)

    return json.dumps({
        "rule_name": rule_name,
        "persona_id": _active_persona,
        "rendered_rule": rendered,
    }, indent=2)


@mcp.tool()
def get_persona_modules(modules: list[str], question_context: str = "") -> str:
    """Selectively retrieve specific persona modules for the active persona.

    This is the core tool implementing selective retrieval. Before answering
    a survey question, the model should identify which modules are relevant
    and retrieve only those, mirroring how real respondents selectively draw
    on relevant memories (Tourangeau et al., 2000).

    Args:
        modules: List of module names to retrieve. Valid modules:
            demographics, life_narrative, health, economy, politics,
            social_views, local_context.
        question_context: Optional brief description of the survey question
            being answered. Used for logging retrieval patterns.
    """
    if _active_persona is None:
        return json.dumps({
            "error": "No active persona. Call set_active_persona first.",
        })

    persona = _personas.get(_active_persona)
    if persona is None:
        return json.dumps({"error": f"Persona '{_active_persona}' not found in database."})

    # Validate requested modules
    invalid = [m for m in modules if m not in VALID_MODULES]
    if invalid:
        return json.dumps({
            "error": f"Invalid module(s): {invalid}",
            "valid_modules": VALID_MODULES,
        })

    # Retrieve requested modules
    retrieved = {}
    for module_name in modules:
        module_data = persona.get(module_name)
        if module_data is not None:
            retrieved[module_name] = module_data
        else:
            retrieved[module_name] = {"note": "Module exists but has no data for this persona."}

    # Log the retrieval pattern for analysis
    log_entry = {
        "persona_id": _active_persona,
        "question_context": question_context,
        "modules_requested": modules,
        "modules_returned": list(retrieved.keys()),
    }
    _retrieval_log.append(log_entry)

    return json.dumps({
        "persona_id": _active_persona,
        "modules_retrieved": modules,
        "data": retrieved,
    }, indent=2)


@mcp.tool()
def get_retrieval_log() -> str:
    """Get the log of all persona module retrievals in the current session.

    This tool supports RQ3 of the framework: analyzing whether retrieval
    patterns correspond to question-relevant memory access as predicted
    by the cognitive model of survey response. Researchers can examine
    which modules the model chose to retrieve for different question domains.
    """
    return json.dumps({
        "session_retrievals": _retrieval_log,
        "total_retrievals": len(_retrieval_log),
    }, indent=2)


@mcp.tool()
def clear_retrieval_log() -> str:
    """Clear the retrieval log. Call between experimental conditions."""
    global _retrieval_log
    _retrieval_log = []
    return json.dumps({"status": "Retrieval log cleared."})


@mcp.tool()
def get_survey_skill(skill_type: str = "general") -> str:
    """Load a survey response skill (structured reasoning procedure).

    Skills correspond to the judgment stage of the cognitive model. They
    instruct the model on how to process each survey question: identify
    relevant personal circumstances, retrieve appropriate modules, and
    select an answer.

    Args:
        skill_type: Type of skill to load. Options:
            "general" - standard survey response procedure
            "sensitive" - for sensitive/controversial topics
            "attitudinal" - for Likert-scale attitude items
    """
    skill_content = _load_skill(skill_type)
    if skill_content is None:
        available = [f.stem for f in SKILLS_DIR.glob("*.txt")] if SKILLS_DIR.exists() else []
        return json.dumps({
            "error": f"Skill '{skill_type}' not found.",
            "available_skills": available,
        })

    return json.dumps({
        "skill_type": skill_type,
        "instructions": skill_content,
    }, indent=2)


@mcp.tool()
def get_framework_status() -> str:
    """Get the current status of all three framework layers.

    Returns the state of Rules (active persona), Skills (available),
    and MCP (retrieval statistics). Useful for verifying that the
    framework is properly configured before running a simulation.
    """
    available_skills = [f.stem for f in SKILLS_DIR.glob("*.txt")] if SKILLS_DIR.exists() else []
    available_rules = [f.stem for f in RULES_DIR.glob("*.txt")] if RULES_DIR.exists() else []

    status = {
        "rules": {
            "active_persona": _active_persona,
            "persona_loaded": _active_persona is not None,
            "available_rule_templates": available_rules,
        },
        "skills": {
            "available": available_skills,
            "count": len(available_skills),
        },
        "mcp": {
            "personas_in_database": len(_personas),
            "available_modules": VALID_MODULES,
            "retrievals_this_session": len(_retrieval_log),
        },
    }
    return json.dumps(status, indent=2)


# ---------------------------------------------------------------------------
# MCP Resources (read-only data the client can browse)
# ---------------------------------------------------------------------------


@mcp.resource("persona://schema")
def get_schema() -> str:
    """Return the persona database schema (module definitions)."""
    schema = {
        "description": "Modular persona database schema for silicon sampling",
        "modules": {
            "demographics": "Age, gender, region, education, income, ethnicity, household composition",
            "life_narrative": "Family history, career trajectory, formative life events, personal milestones",
            "health": "Insurance status, chronic conditions, health behaviors, healthcare experiences",
            "economy": "Employment situation, financial anxieties, economic outlook, class identity",
            "politics": "Party identification, past voting behavior, political engagement, media diet",
            "social_views": "Stances on immigration, race, gender, religion, cultural issues",
            "local_context": "State/regional facts, local news awareness, community characteristics",
        },
        "design_principles": [
            "Each module is self-contained and can be retrieved independently",
            "Modules map to distinct domains of survey questions",
            "Granularity allows the model to perform selective retrieval per question",
            "Schema is extensible: researchers can add domain-specific modules",
        ],
    }
    return json.dumps(schema, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silicon Sampling MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)
