"""
Deletes all Zendesk tickets logged during testing.

Run:
    python cleanup_test_tickets.py
"""

import json
from dotenv import load_dotenv
from zendesk_utils import delete_logged_tickets, fetch_ticket, TEST_LOG

load_dotenv()

try:
    with open(TEST_LOG, encoding="utf-8") as f:
        log = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    print("No test tickets logged. Nothing to delete.")
    raise SystemExit

if not log:
    print("test_tickets.json is empty. Nothing to delete.")
    raise SystemExit

print(f"Found {len(log)} logged ticket(s):\n")
for idx, entry in enumerate(log):
    print(f"  [{idx + 1}] Ticket #{entry['id']} — submitted at {entry['submitted_at']}")

while True:
    print("\nOptions:")
    print("  Enter a ticket number (e.g. 1) to inspect it")
    print("  'd' to delete all and exit")
    print("  'q' to quit without deleting")

    choice = input("\n> ").strip().lower()

    if choice == "q":
        print("Aborted. No tickets deleted.")
        break

    if choice == "d":
        confirm = input(f"\nDelete all {len(log)} ticket(s) from Zendesk? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            break

        deleted, failed = delete_logged_tickets()
        if deleted:
            print(f"\nDeleted: {deleted}")
        if failed:
            print(f"Failed to delete (still in log): {failed}")
        if not failed:
            print("test_tickets.json cleared.")
        break

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(log):
            ticket_id = log[idx]["id"]
            print(f"\nFetching Ticket #{ticket_id} ...")
            success, data = fetch_ticket(ticket_id)
            if success:
                print(f"  ID          : {data['id']}")
                print(f"  Subject     : {data['subject']}")
                print(f"  Status      : {data['status']}")
                print(f"  Created at  : {data['created_at']}")
                print(f"  Tags        : {', '.join(data.get('tags', []))}")
                print(f"  Description :\n")
                for line in data.get("description", "").splitlines():
                    print(f"    {line}")
            else:
                print(f"  Failed to fetch: {data}")
        else:
            print("  Invalid selection.")
    else:
        print("  Unrecognised input.")
