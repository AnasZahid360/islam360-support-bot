"""
Zendesk integration — creates support tickets via the Zendesk REST API.

Requires in .env:
    ZENDESK_SUBDOMAIN  e.g. islam360  (for islam360.zendesk.com)
    ZENDESK_EMAIL      agent/admin email used to authenticate
    ZENDESK_API_TOKEN  API token from Admin > Apps & Integrations > APIs > Zendesk API
"""

import os
import json
import requests
from datetime import datetime


def _base_url() -> str:
    subdomain = os.getenv("ZENDESK_SUBDOMAIN", "")
    # Accept either full URL or just the subdomain
    subdomain = subdomain.replace("https://", "").replace("http://", "").rstrip("/")
    if ".zendesk.com" in subdomain:
        return f"https://{subdomain}/api/v2"
    return f"https://{subdomain}.zendesk.com/api/v2"


def _auth() -> tuple[str, str]:
    email = os.getenv("ZENDESK_EMAIL")
    token = os.getenv("ZENDESK_API_TOKEN")
    return (f"{email}/token", token)


def _upload_file(file_bytes: bytes, filename: str, content_type: str) -> str | None:
    """
    Uploads a file to Zendesk and returns the upload token.
    Returns None if the upload fails.
    """
    response = requests.post(
        f"{_base_url()}/uploads.json?filename={filename}",
        auth=_auth(),
        headers={"Content-Type": content_type},
        data=file_bytes,
    )
    if response.status_code == 201:
        return response.json()["upload"]["token"]
    return None


def post_ticket(
    title: str,
    description: str,
    conversation_context: str = "",
    screenshot_bytes: bytes | None = None,
    screenshot_filename: str | None = None,
    screenshot_content_type: str | None = None,
) -> tuple[bool, str]:
    """
    Creates a Zendesk ticket, optionally attaching a screenshot.

    Returns:
        (success: bool, ticket_id or error_message)
    """
    subdomain = os.getenv("ZENDESK_SUBDOMAIN")
    email     = os.getenv("ZENDESK_EMAIL")
    token     = os.getenv("ZENDESK_API_TOKEN")

    if not all([subdomain, email, token]):
        return False, "ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, or ZENDESK_API_TOKEN not set in .env"

    # Build ticket body
    body_parts = [description]
    if conversation_context:
        body_parts.append(f"\n---\nConversation context:\n{conversation_context}")
    full_body = "\n".join(body_parts)

    # Upload screenshot if provided
    upload_tokens = []
    if screenshot_bytes and screenshot_filename:
        upload_token = _upload_file(
            screenshot_bytes,
            screenshot_filename,
            screenshot_content_type or "image/png",
        )
        if upload_token:
            upload_tokens.append(upload_token)

    comment: dict = {"body": full_body}
    if upload_tokens:
        comment["uploads"] = upload_tokens

    payload: dict = {
        "ticket": {
            "subject": title,
            "comment": comment,
            "tags": ["bot-submitted"],
        }
    }

    response = requests.post(
        f"{_base_url()}/tickets.json",
        auth=_auth(),
        json=payload,
    )

    if response.status_code == 201:
        ticket_id = response.json()["ticket"]["id"]
        _log_ticket_id(ticket_id)
        return True, str(ticket_id)
    else:
        return False, f"{response.status_code} — {response.text}"


def fetch_ticket(ticket_id: int) -> tuple[bool, dict]:
    """
    Fetches full details of a single Zendesk ticket.

    Returns:
        (success: bool, ticket_dict or error_str)
    """
    response = requests.get(
        f"{_base_url()}/tickets/{ticket_id}.json",
        auth=_auth(),
    )
    if response.status_code == 200:
        return True, response.json()["ticket"]
    return False, response.text


# ---------------------------------------------------------------------------
# Test ticket tracking
# ---------------------------------------------------------------------------
TEST_LOG = "test_tickets.json"


def _log_ticket_id(ticket_id: int) -> None:
    """Appends a submitted ticket ID to the local test log."""
    try:
        with open(TEST_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = []

    log.append({"id": ticket_id, "submitted_at": datetime.utcnow().isoformat()})

    with open(TEST_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def delete_logged_tickets() -> tuple[list[int], list[int]]:
    """
    Reads test_tickets.json and bulk-deletes all logged tickets from Zendesk.

    Returns:
        (deleted_ids, failed_ids)
    """
    try:
        with open(TEST_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], []

    ids = [entry["id"] for entry in log]
    if not ids:
        return [], []

    # Zendesk bulk delete: up to 100 ids per request
    deleted, failed = [], []
    chunk_size = 100
    for offset in range(0, len(ids), chunk_size):
        chunk = ids[offset : offset + chunk_size]
        ids_param = ",".join(str(tid) for tid in chunk)
        response = requests.delete(
            f"{_base_url()}/tickets/destroy_many.json?ids={ids_param}",
            auth=_auth(),
        )
        if response.status_code in (200, 204):
            deleted.extend(chunk)
        else:
            failed.extend(chunk)

    # Clear the log for successfully deleted tickets
    if failed:
        remaining = [e for e in log if e["id"] in failed]
    else:
        remaining = []

    with open(TEST_LOG, "w", encoding="utf-8") as f:
        json.dump(remaining, f, indent=2)

    return deleted, failed
