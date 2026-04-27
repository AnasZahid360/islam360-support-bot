# Islam360 Zendesk Support Bot

This project builds a retrieval-based customer support assistant for Islam360 using historical Zendesk tickets, then provides a testable frontend to chat with it.

## What this repo does

- Converts Zendesk exports into RAG-ready JSONL records.
- Builds a FAISS vector index from answered support conversations.
- Answers user questions using retrieval + LLM generation.
- Escalates unresolved issues by creating Zendesk tickets.
- Provides two frontends for testing:
  - Streamlit app (`app.py`)
  - FastAPI + web chat UI (`api.py` + `test_ui.html`)

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with:

```env
OPENAI_API_KEY=...
ZENDESK_SUBDOMAIN=...
ZENDESK_EMAIL=...
ZENDESK_API_TOKEN=...
```

## Build the RAG data + index

```bash
python prepare_rag.py
python build_index.py
python add_knowledge.py
```

This produces/updates `rag_documents.jsonl` and `faiss_index/`.

## Frontend option 1: FastAPI test UI (recommended for quick testing)

```bash
python api.py
```

Open: `http://localhost:8000`

Health endpoint: `http://localhost:8000/health`

## Frontend option 2: Streamlit support UI

```bash
streamlit run app.py
```

## Useful scripts

- `rag_query.py` - CLI chat against the same RAG pipeline.
- `cleanup_test_tickets.py` - inspect/delete tickets created during testing.