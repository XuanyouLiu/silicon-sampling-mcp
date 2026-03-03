# Silicon Sampling MCP Server Demo

A Model Context Protocol (MCP) server implementing **selective persona retrieval** for silicon sampling research. This demo accompanies the paper:

> *Beyond Prompting: A Cognitive-Grounded Framework for Silicon Sampling with Rules, Skills, and Model Context Protocol (MCP)*

## Overview

![Framework Architecture](framework.png)

This server operationalizes the cognitive model of survey response (Tourangeau et al., 2000) by storing persona information in **modular databases** and allowing LLMs to **selectively retrieve** only the modules relevant to each survey question. The three layers of the framework map to:

| Layer | Role (cognitive stage) | Implementation |
|-------|-------------------------|----------------|
| **Rules** | Comprehension: identity anchor (age, gender, race, education, region) & constraints | `set_active_persona` + `get_rule`; researcher-specified |
| **Skills** | Judgment: question-type-specific reasoning (general, sensitive-topic, attitudinal) | `get_survey_skill`; model selects per question |
| **MCP** | Retrieval: selective access to modular persona DB | `get_persona_modules`; model decides which modules to retrieve |

## Quick Start

### Prerequisites

- Python 3.10+
- `pip install mcp` (or `pip install "mcp[cli]"`)

### Installation

```bash
cd demo
pip install -r requirements.txt
```

### Run the Server

```bash
# stdio transport (for Cursor, Claude Desktop, etc.)
python server.py

# SSE transport (for web clients)
python server.py --transport sse
```

### Configure in Cursor

Add to your `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "silicon-sampling": {
      "command": "python",
      "args": ["/absolute/path/to/demo/server.py"]
    }
  }
}
```

### Configure in Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "silicon-sampling": {
      "command": "python",
      "args": ["/absolute/path/to/demo/server.py"]
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `list_personas` | List all personas in the database |
| `set_active_persona` | Activate a persona (Rules layer) |
| `get_rule` | Load a Rule template (survey_respondent, baseline_static, rules_only) for ablation |
| `get_persona_modules` | Selectively retrieve persona modules (MCP layer) |
| `get_survey_skill` | Load a reasoning skill (general, sensitive, attitudinal) |
| `get_retrieval_log` | View all module retrievals and skill selections this session |
| `clear_retrieval_log` | Reset the retrieval log between conditions |
| `get_framework_status` | Check status of all three layers |

## Persona Database Schema

Each persona is a JSON file in `personas/` with seven modules:

| Module | Content (per extended abstract Table 1) |
|--------|----------------------------------------|
| `demographics` | Age, gender, region, education, income, ethnicity |
| `life_narrative` | Family structure, career trajectory, formative events |
| `health` | Insurance status, care experiences, views on healthcare policy |
| `economy` | Employment history, financial stressors, views on economic policy |
| `politics` | Party ID, voting history, views on government role |
| `social_context` | Intergroup contact, religious practice, views on social issues |
| `local_context` | State-level facts, local news, community characteristics |

## Example Usage

A typical survey simulation session:

```
1. list_personas                              # Browse available respondents
2. set_active_persona("persona_001")          # Activate Linda Kowalski (Rules)
3. get_survey_skill("general")                # Load general survey skill (Skills)
4. get_persona_modules(                       # Selective retrieval (MCP)
       modules=["health", "economy", "demographics"],
       question_context="government healthcare spending"
   )
5. [Model answers the survey question using retrieved information]
6. get_retrieval_log()                        # Analyze retrieval patterns
```

## Example Personas

| ID | Name | Age | State | Education | Party |
|----|------|-----|-------|-----------|-------|
| `persona_001` | Linda Kowalski | 52 | OH | Associate degree | Ind. (lean D) |
| `persona_002` | Marcus Williams | 34 | GA | Bachelor's (IT) | Democrat |
| `persona_003` | Ryan Dawson | 28 | TX | Some college | Republican |

## Skills

| Skill | Use Case (judgment stage: model selects per question) |
|-------|--------------------------------------------------------|
| `general` | Standard survey response procedure |
| `sensitive` | Sensitive-topic questions (race, immigration, sexuality, religion, income) |
| `attitudinal` | Likert-scale and feeling thermometer items |

## Adding Your Own Personas

Create a new JSON file in `personas/` following the schema above. The server auto-discovers all `.json` files in the directory on startup.

## Ablation Conditions (per extended abstract)

The evaluation plan compares four conditions: (a) static backstory baseline (`baseline_static` rule); (b) Rules only (`rules_only` rule, no Skills or MCP); (c) Rules + Skills (full rule + `get_survey_skill`, no MCP retrieval); (d) full framework with MCP-based selective retrieval (`survey_respondent` rule + Skills + `get_persona_modules`).

## Research Applications

- **Ablation studies**: Compare the four conditions above to isolate each layer's contribution
- **Retrieval analysis**: Use `get_retrieval_log` to study which modules the model accesses per question domain
- **Bias measurement**: Compare response distributions across conditions
- **Cross-model testing**: Run the same server with different LLM backends

## License

MIT
