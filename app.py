"""
Islam360 Customer Support Bot — Streamlit frontend.

Run:
    streamlit run app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from rag_query import ask, check_ticket_needed
from zendesk_utils import post_ticket

# Load secrets — st.secrets when on Streamlit Cloud, .env locally
load_dotenv()
_SECRET_KEYS = ["OPENAI_API_KEY", "ZENDESK_SUBDOMAIN", "ZENDESK_EMAIL", "ZENDESK_API_TOKEN"]
for _key in _SECRET_KEYS:
    if _key not in os.environ and hasattr(st, "secrets") and _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Islam360 Support",
    page_icon="☪️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []        # LangChain HumanMessage/AIMessage list
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []    # {"role": "user"/"agent", "content": str, "image": bytes|None}
if "ticket_created" not in st.session_state:
    st.session_state.ticket_created = False
if "ticket_id" not in st.session_state:
    st.session_state.ticket_id = None
if "screenshot" not in st.session_state:
    st.session_state.screenshot = None        # {"bytes", "name", "type"} — saved for ticket attachment

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("☪️ Islam360 Customer Support")
st.caption("Assalamualaikum! How can we help you today?")

# ---------------------------------------------------------------------------
# Render existing chat messages
# ---------------------------------------------------------------------------
for msg in st.session_state.display_messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        if msg.get("image"):
            st.image(msg["image"], width=260)
        if msg.get("content"):
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input — native file attachment (paperclip inside the input bar)
# ---------------------------------------------------------------------------
response = st.chat_input(
    "Type your message here...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp", "gif"],
)

# ---------------------------------------------------------------------------
# Handle send
# ---------------------------------------------------------------------------
if response:
    user_input = response.text or ""
    attached = response.files[0] if response.files else None

    # Must have at least a text message
    if not user_input.strip() and not attached:
        st.stop()

    # Read image bytes if attached
    pending: dict | None = None
    if attached:
        pending = {
            "bytes": attached.getvalue(),
            "name": attached.name,
            "type": attached.type,
        }
        # Save first screenshot of the session for the ticket
        if st.session_state.screenshot is None:
            st.session_state.screenshot = pending

    # Build and show user message bubble
    display_msg: dict = {"role": "user", "content": user_input}
    if pending:
        display_msg["image"] = pending["bytes"]

    st.session_state.display_messages.append(display_msg)

    with st.chat_message("user"):
        if pending:
            st.image(pending["bytes"], width=260)
        if user_input:
            st.markdown(user_input)

    # Get RAG answer
    with st.chat_message("assistant"):
        with st.spinner("..."):
            answer, _, st.session_state.chat_history = ask(
                user_input or "[User attached a screenshot]",
                st.session_state.chat_history,
            )
        st.markdown(answer)

    st.session_state.display_messages.append({"role": "agent", "content": answer})

    # Check if a Zendesk ticket should be filed (only once per session)
    if not st.session_state.ticket_created:
        decision = check_ticket_needed(st.session_state.chat_history)

        if decision.get("create"):
            screenshot = st.session_state.screenshot
            success, result = post_ticket(
                title=decision["title"],
                description=decision["description"],
                conversation_context="\n".join(
                    f"{'User' if not isinstance(m, AIMessage) else 'Agent'}: {m.content}"
                    for m in st.session_state.chat_history
                ),
                screenshot_bytes=screenshot["bytes"] if screenshot else None,
                screenshot_filename=screenshot["name"] if screenshot else None,
                screenshot_content_type=screenshot["type"] if screenshot else None,
            )

            if success:
                st.session_state.ticket_created = True
                st.session_state.ticket_id = result
                screenshot_note = " Your screenshot has been attached." if screenshot else ""
                ticket_msg = (
                    f"*(Your issue has been logged with our team — Ticket #{result}.{screenshot_note} "
                    "We'll follow up with you shortly. JazakAllah Khair!)*"
                )
                st.session_state.display_messages.append({"role": "agent", "content": ticket_msg})
                st.session_state.chat_history.append(AIMessage(content=ticket_msg))
                with st.chat_message("assistant"):
                    st.markdown(ticket_msg)

    st.rerun()
