# Agents Configuration

## md-agent

### Role
Manages and organizes markdown-based research knowledge.

### Scope
- Reads all files under `notes/`
- Focus areas:
  - concepts
  - graphics
  - models
  - papers
  - meetings

### Responsibilities

1. Knowledge Structuring
- Convert unstructured notes into structured concept-based notes
- Ensure each concept has a dedicated file in `notes/concepts/`

2. Linking
- Create bidirectional links using [[...]]
- Connect:
  - concepts ↔ papers
  - concepts ↔ experiments
  - meetings ↔ concepts

3. Normalization
- Remove duplicate or overlapping notes
- Merge similar concepts

4. Extraction
From meeting notes:
- extract problems
- extract research directions
- extract keywords → convert into concept notes if missing

5. Constraints
- Do NOT modify files under `projects/`
- Do NOT write code
- Only modify markdown files

### Output Rules
- Always update or create `.md` files
- Prefer updating existing notes over creating duplicates