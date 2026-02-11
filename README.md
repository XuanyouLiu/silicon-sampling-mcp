# Silicon Sampling MCP Server Demo

A Model Context Protocol (MCP) server implementing **selective persona retrieval** for silicon sampling research. This demo accompanies the paper:

> *Beyond Prompting: A Normative Framework for Silicon Sampling with Rules, Skills, and MCP*

## Overview

This server operationalizes the cognitive model of survey response (Tourangeau et al., 2000) by storing persona information in **modular databases** and allowing LLMs to **selectively retrieve** only the modules relevant to each survey question. The three layers of the framework map to:

| Layer | Role | Implementation |
|-------|------|----------------|
| **Rules** | Identity anchor & constraints | `set_active_persona` sets the respondent identity |
| **Skills** | Structured reasoning procedures | `get_survey_skill` loads question-type-specific reasoning |
| **MCP** | Selective persona retrieval | `get_persona_modules` retrieves specific modules on demand |

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
| `get_persona_modules` | Selectively retrieve persona modules (MCP layer) |
| `get_survey_skill` | Load a reasoning skill template (Skills layer) |
| `get_retrieval_log` | View all module retrievals this session |
| `clear_retrieval_log` | Reset the retrieval log between conditions |
| `get_framework_status` | Check status of all three layers |

## Persona Database Schema

Each persona is a JSON file in `personas/` with seven modules:

| Module | Content |
|--------|---------|
| `demographics` | Age, gender, region, education, income, ethnicity |
| `life_narrative` | Family, career trajectory, formative events |
| `health` | Insurance, conditions, health behaviors |
| `economy` | Employment, financial situation, economic views |
| `politics` | Party ID, voting history, political engagement |
| `social_views` | Stances on immigration, race, gender, religion |
| `local_context` | State-level facts, local news, community traits |

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

| Skill | Use Case |
|-------|----------|
| `general` | Standard survey response procedure |
| `sensitive` | Race, immigration, sexuality, income questions |
| `attitudinal` | Likert-scale and feeling thermometer items |

## Adding Your Own Personas

Create a new JSON file in `personas/` following the schema above. The server auto-discovers all `.json` files in the directory on startup.

## Research Applications

- **Ablation studies**: Compare conditions by toggling Rules, Skills, and MCP
- **Retrieval analysis**: Use `get_retrieval_log` to study which modules the model accesses per question domain
- **Bias measurement**: Compare response distributions across conditions
- **Cross-model testing**: Run the same server with different LLM backends

## License

MIT
