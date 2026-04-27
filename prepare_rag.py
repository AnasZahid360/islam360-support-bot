"""
Converts all Zendesk training data into rag_documents.jsonl for RAG.

Sources processed:
  - zendesk-training-data-2026-03-31.xlsx  (Oct 2025 – Mar 2026)
  - zendesk-training-data-2022-01-01_to_2025-10-01.csv  (Mar 2023 – Sep 2025)

Output: rag_documents.jsonl
  - One JSON object per line, one per ticket (deduped by Ticket ID).
  - `conversation` field is the primary embed target (full thread as plain text).
  - `query` / `response` fields for structured Q&A access.
  - Tickets without any agent reply are included with has_response=false
    (skipped during indexing but kept for gap analysis).

Usage:
    python prepare_rag.py
"""

import json
import re
import pandas as pd
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# Spam subjects to filter out (system-generated, not customer conversations)
# ---------------------------------------------------------------------------
SPAM_PATTERNS = [
    "DAILY FAILED TXN",
    "DAILY FAILED TXN was executed",
]


def is_spam(subject: str) -> bool:
    if not isinstance(subject, str):
        return False
    return any(p.lower() in subject.lower() for p in SPAM_PATTERNS)


# ---------------------------------------------------------------------------
# HTML stripper
# ---------------------------------------------------------------------------
class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "<" not in text:
        return text.strip()
    s = _HTMLStripper()
    s.feed(text)
    raw = " ".join(s.parts)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


# ---------------------------------------------------------------------------
# Load all sources into a single normalised DataFrame
# ---------------------------------------------------------------------------
SOURCES = [
    ("zendesk-training-data-2026-03-31.xlsx", "xlsx"),
    ("zendesk-training-data-2022-01-01_to_2025-10-01.csv", "csv"),
]
OUTPUT_FILE = "rag_documents.jsonl"


def load_sources() -> pd.DataFrame:
    frames = []
    for path, fmt in SOURCES:
        print(f"  Loading {path} ...")
        try:
            df = pd.read_excel(path) if fmt == "xlsx" else pd.read_csv(path)
        except FileNotFoundError:
            print(f"    WARNING: {path} not found, skipping.")
            continue
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
        print(f"    {len(df):,} rows loaded")

    combined = pd.concat(frames, ignore_index=True)

    # Drop the one null Author Role row
    combined = combined.dropna(subset=["Author Role"])

    # Sort within each ticket by message order
    combined = combined.sort_values(["Ticket ID", "Message #"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Build one record per ticket
# ---------------------------------------------------------------------------
def build_records(df: pd.DataFrame) -> list[dict]:
    records = []
    skipped_empty = 0
    skipped_spam = 0
    seen_ids = set()

    for ticket_id, group in df.groupby("Ticket ID", sort=False):
        # Deduplicate — keep first occurrence if same ticket appears in both sources
        if ticket_id in seen_ids:
            continue
        seen_ids.add(ticket_id)

        subject = str(group["Subject"].iloc[0]).strip()
        created_date = str(group["Created Date"].iloc[0])

        # Filter spam
        if is_spam(subject):
            skipped_spam += 1
            continue

        customer_msgs = []
        agent_msgs = []
        conversation_lines = []

        for _, row in group.iterrows():
            role = str(row["Author Role"]).strip().lower()
            body = strip_html(str(row["Message Body"]))

            if not body or body.lower() in ("nan", "none", ""):
                continue

            if role == "customer":
                customer_msgs.append(body)
                conversation_lines.append(f"[Customer]: {body}")
            elif role == "agent":
                agent_msgs.append(body)
                conversation_lines.append(f"[Agent]: {body}")
            else:
                conversation_lines.append(f"[{role}]: {body}")

        if not customer_msgs:
            skipped_empty += 1
            continue

        conversation_text = f"Subject: {subject}\n\n" + "\n\n".join(conversation_lines)

        records.append({
            "id": str(ticket_id),
            "subject": subject,
            "created_date": created_date,
            "query": "\n\n".join(customer_msgs),
            "response": "\n\n".join(agent_msgs) if agent_msgs else None,
            "has_response": bool(agent_msgs),
            "conversation": conversation_text,
        })

    print(f"  Filtered out — spam       : {skipped_spam:,}")
    print(f"  Filtered out — empty/null : {skipped_empty:,}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading sources ...")
    df = load_sources()
    print(f"  Combined rows: {len(df):,} | Unique tickets: {df['Ticket ID'].nunique():,}\n")

    print("Processing tickets ...")
    records = build_records(df)

    answered = sum(1 for r in records if r["has_response"])
    unanswered = len(records) - answered

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. Written to {OUTPUT_FILE}")
    print(f"  Total documents : {len(records):,}")
    print(f"  With agent reply: {answered:,}")
    print(f"  No agent reply  : {unanswered:,}")


if __name__ == "__main__":
    main()
