# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **data project**, not a software application. The primary artifact is a Zendesk customer support ticket export intended for AI training, analytics, or knowledge base generation for the Islam360 ecosystem.

**Main file:** `zendesk-training-data-2026-03-31.xlsx`

## Data Structure

The Excel file contains one sheet with 5,547 rows and these columns:

| Column | Description |
|---|---|
| Ticket ID | Zendesk ticket identifier |
| Created Date | Ticket creation timestamp |
| Subject | Ticket subject line |
| Message # | Sequential message index within the ticket thread |
| Author Role | `customer` or `agent` |
| Author Name | Name of the message author |
| Message Time | Timestamp of the individual message |
| Message Body | Full message text (may include HTML, multilingual content) |

**Languages:** English, Urdu, Arabic
**Content types:** Technical support, feature requests, collaboration inquiries, general questions

## RAG Output

**`rag_documents.jsonl`** — primary artifact for vector store ingestion. One JSON object per line, one per ticket.

| Field | Description |
|---|---|
| `id` | Ticket ID (string) |
| `subject` | Ticket subject |
| `created_date` | Ticket creation timestamp |
| `query` | All customer message(s) joined — what gets matched at retrieval time |
| `response` | All agent message(s) joined — the answer to surface (`null` if no reply) |
| `has_response` | Boolean — filter to `true` for answered-only indexing |
| `conversation` | Full thread as plain text (`Subject:` + `[Customer]:` / `[Agent]:` turns) — **the field to embed** |

Stats: 3,866 documents total — 1,241 with agent replies, 2,625 customer-only.

Regenerate from source: `python prepare_rag.py`

**Recommended indexing strategy:** embed `conversation`, filter on `has_response=true` for production RAG, keep unanswered tickets separately to identify coverage gaps.

## Working with the Data

The XLSX file is a ZIP archive. To inspect its raw XML without Python:

```bash
# List archive contents
unzip -l zendesk-training-data-2026-03-31.xlsx

# Read sheet data
unzip -p zendesk-training-data-2026-03-31.xlsx xl/worksheets/sheet1.xml

# Read shared strings (most text content lives here)
unzip -p zendesk-training-data-2026-03-31.xlsx xl/sharedStrings.xml
```

To load and process via Python:

```python
import pandas as pd
df = pd.read_excel("zendesk-training-data-2026-03-31.xlsx")
```

## Broader Context

This project is part of the Islam360 ecosystem. Related projects on the same machine include an agentic AI system (`ISLAM360 AGENTIC`), a chatbot (`ISLAM360 CHATBOT`), and a dashboard (`Islam360-Dashboard-dev-360`). This dataset likely feeds into one or more of those applications as training data.
