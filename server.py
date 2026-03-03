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
    python server.py --allowed-modules demographics politics economy
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

PERSONAS_DIR = Path(__file__).parent / "personas"
SKILLS_DIR = Path(__file__).parent / "skills"
RULES_DIR = Path(__file__).parent / "rules"

VALID_MODULES = [
    "demographics", "life_narrative", "politics", "economy",
    "health", "social_context", "racial_attitudes",
    "values_personality", "media_consumption",
    "religion_community", "local_context",
    "policy_positions", "civic_participation",
]

mcp = FastMCP(
    "Silicon Sampling Server",
    instructions=(
        "MCP server for selective persona retrieval in silicon sampling. "
        "Implements modular persona databases that LLMs can query per-question."
    ),
)

_personas: dict[str, dict] = {}
_retrieval_log: list[dict] = []
_skill_log: list[dict] = []
_allowed_modules: Optional[set[str]] = None


def _load_personas() -> None:
    global _personas
    _personas = {}
    if not PERSONAS_DIR.exists():
        return
    for fp in sorted(PERSONAS_DIR.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            _personas[fp.stem] = json.load(f)


_load_personas()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_survey_skill(skill_type: str, question_context: str = "") -> str:
    """Select and load a survey response Skill (reasoning procedure).

    Call this FIRST after reading each survey question. Choose the Skill
    that matches the question type:

      "factual_recall" - about YOUR personal situation or circumstances.
          e.g., "Are you better off financially?", "Has the economy
          gotten better or worse?", "How worried are you about money?"

      "direct_attitude" - about a SINGLE social/policy topic where you
          have a position. e.g., "Do you favor the death penalty?",
          "Should gun access be easier or harder?", "Do you agree that
          newer lifestyles are contributing to breakdown?"

      "attitude_construction" - about a COMPLEX topic spanning multiple
          life areas. e.g., "How much can you trust the government?",
          "Should the government provide more services?", "Do you favor
          the Affordable Care Act?", "Should immigration increase?"

    Args:
        skill_type: One of "factual_recall", "direct_attitude", "attitude_construction".
        question_context: Brief description of the survey question.
    """
    skill_path = SKILLS_DIR / f"{skill_type}.txt"
    if not skill_path.exists():
        available = [f.stem for f in SKILLS_DIR.glob("*.txt")] if SKILLS_DIR.exists() else []
        return json.dumps({"error": f"Skill '{skill_type}' not found.", "available": available})

    _skill_log.append({"question": question_context, "skill": skill_type})

    return json.dumps({
        "skill_type": skill_type,
        "instructions": skill_path.read_text(encoding="utf-8"),
    }, indent=2)


@mcp.tool()
def get_persona_modules(persona_id: str, modules: list[str], question_context: str = "") -> str:
    """Selectively retrieve specific persona modules.

    Call this AFTER selecting a Skill. Retrieve ONLY modules relevant to
    the current survey question. Real respondents do not scan their entire
    autobiography; they selectively recall information relevant to the
    question at hand.

    Available modules (13 total, ~170 fields per persona):
      demographics       - Age, gender, race, education, income, marital status, religion, state, region
      life_narrative     - Summary of life circumstances
      politics           - Party ID, ideology, approval, voting behavior, candidate trait evaluations, election legitimacy, democratic satisfaction, feelings toward political entities, issue positions, participation
      economy            - Employment, income, housing, investments, food security, bills, economic outlook, trade views, feelings toward economic groups
      health             - Self-reported health, insurance, healthcare concerns, mental health, life satisfaction, medicare/drug pricing views, 10 health conditions
      social_context     - Social trust, PC sensitivity, violence views, feelings toward social groups, immigration positions, police funding, transgender policy, affirmative action, gun permits
      racial_attitudes   - Feelings toward racial/ethnic groups, discrimination perceptions for multiple groups, racial equality views, race-related policy views
      values_personality - Moral foundations, authoritarianism, science attitudes, environment-economy tradeoff, egalitarianism
      media_consumption  - News sources, social media use and hours, Fox/CNN use, political news interest, feelings toward institutions
      religion_community - Religious tradition, attendance, importance, children, community ties, loneliness
      local_context      - State, region
      policy_positions   - Government spending priorities (social security, schools, crime, childcare, aid to poor, environment, border security, infrastructure), candidate issue placements, candidate competence ratings
      civic_participation - Feelings toward US and UN, campaign volunteering, signs, buttons

    Retrieve only the modules relevant to the current question. Do NOT retrieve all modules.

    Args:
        persona_id: The ID of the persona (e.g., "anes_001").
        modules: List of module names to retrieve.
        question_context: Brief description of the survey question.
    """
    if not _personas:
        _load_personas()

    persona = _personas.get(persona_id)
    if persona is None:
        return json.dumps({"error": f"Persona '{persona_id}' not found.", "available": list(_personas.keys())})

    if _allowed_modules is not None:
        effective_modules = [m for m in modules if m in _allowed_modules]
        unavailable = [m for m in modules if m not in _allowed_modules]
    else:
        effective_modules = modules
        unavailable = []

    invalid = [m for m in effective_modules if m not in VALID_MODULES]
    if invalid:
        return json.dumps({"error": f"Invalid module(s): {invalid}", "valid_modules": VALID_MODULES})

    retrieved = {}
    for mod in effective_modules:
        data = persona.get(mod)
        if data is not None:
            retrieved[mod] = data
        else:
            retrieved[mod] = {"note": "No data available for this module."}

    for mod in unavailable:
        retrieved[mod] = {"note": "Module not available in this experimental phase."}

    _retrieval_log.append({
        "persona_id": persona_id,
        "question": question_context,
        "modules_requested": modules,
        "modules_returned": effective_modules,
    })

    return json.dumps({
        "persona_id": persona_id,
        "modules_retrieved": effective_modules,
        "data": retrieved,
    }, indent=2)


@mcp.tool()
def get_retrieval_log() -> str:
    """Get the log of all Skill selections and module retrievals this session."""
    return json.dumps({
        "skill_selections": _skill_log,
        "module_retrievals": _retrieval_log,
        "total_skills": len(_skill_log),
        "total_retrievals": len(_retrieval_log),
    }, indent=2)


@mcp.resource("persona://schema")
def get_schema() -> str:
    """Return the persona database schema (module definitions)."""
    schema = {
        "modules": {
            "demographics": "Age, gender, race, education, income, marital status, religion, state, region",
            "life_narrative": "Summary of life circumstances",
            "politics": "Party ID, ideology, approval, candidate feeling thermometers, participation, defense/jobs positions",
            "economy": "Employment status, income, housing, investments, food security, bills, economic outlook, trade views, feelings toward economic groups",
            "health": "Insurance, healthcare concerns, mental health, 10 diagnosed health conditions",
            "social_context": "Social trust, PC sensitivity, violence views, feelings toward social groups (LGBTQ, Muslims, etc.), immigration positions",
            "racial_attitudes": "Feelings toward racial/ethnic groups, discrimination perceptions, race-related policy views",
            "values_personality": "Moral foundations (harm, fairness, loyalty, authority, purity), authoritarianism, science attitudes, environment-economy tradeoff",
            "media_consumption": "News sources (TV, newspaper, radio, internet, social media), political news interest, feelings toward institutions (scientists, media, police, military)",
            "religion_community": "Religious tradition, attendance, importance, children, community ties",
            "local_context": "State, region",
        },
    }
    return json.dumps(schema, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silicon Sampling MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument(
        "--allowed-modules", nargs="+", default=None,
        help="Restrict retrievable modules for phase-based experiments",
    )
    args = parser.parse_args()

    if args.allowed_modules:
        _allowed_modules = set(args.allowed_modules)

    mcp.run(transport=args.transport)
