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
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — light background, Islam360 green theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Light background for main area */
.stApp {
    background-color: #f5f5f5;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #1B5E20;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #c8e6c9 !important;
    font-size: 0.75rem;
}

/* Test case buttons */
div[data-testid="stSidebar"] .stButton > button {
    background-color: #2E7D32;
    color: white !important;
    border: 1px solid #4CAF50;
    border-radius: 8px;
    text-align: left;
    width: 100%;
    padding: 8px 12px;
    margin-bottom: 4px;
    font-size: 0.82rem;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #388E3C;
    border-color: #81C784;
}

/* New Conversation button */
div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #4CAF50 !important;
    font-weight: bold;
    font-size: 0.9rem;
}

/* Header bar */
.header-bar {
    background-color: #1B5E20;
    color: white;
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.header-bar h2 {
    margin: 0;
    color: white;
    font-size: 1.3rem;
}
.header-bar small {
    color: #c8e6c9;
    font-size: 0.85rem;
}

/* Chat area */
.chat-area {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    min-height: 400px;
}

/* Badge colors for test cases */
.badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: bold;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Quick test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    {"id": "TC-01", "name": "Subscription — Ads Still Showing",
     "query": "I purchased monthly subscription yesterday but still when I open app to read quran, I get to see ads"},
    {"id": "TC-02", "name": "Font / Display Issue",
     "query": "During the Quran reading some words are not clear on the screen, some words are cut in half."},
    {"id": "TC-03", "name": "Inappropriate Ad",
     "query": "Please remove this ad. It shows things that are hurtful and inappropriate, especially in front of the Quran."},
    {"id": "TC-04", "name": "Quran Content Mistake",
     "query": "Assalamualaikum. There is a mistake in the Quran in the app. There is a missing zabar on alif which changes the pronunciation completely."},
    {"id": "TC-05", "name": "Web / Desktop App Request",
     "query": "Assalamualaikum Islam360 Staff. Do you have a desktop or web application? I spend 8-9 hours at work on a laptop and would love to use Islam360 there."},
    {"id": "TC-06", "name": "Arabic Hadith Search",
     "query": "I have an issue with search. I want to search for any Arabic word or sentence in Hadith books but it never gives results even when the word is definitely in the book."},
    {"id": "TC-07", "name": "Feature — Add Shia Books",
     "query": "Assalamualaikum. Can you please add Shia books to the Islam360 app as well?"},
    {"id": "TC-08", "name": "Donation / IBAN Request",
     "query": "Assalam o alaikum. I want to donate to Islam360. Can you give me the account number and IBAN for overseas donations?"},
    {"id": "TC-09", "name": "Prayer Notifications Not Working",
     "query": "Salam. I am unable to get Namaz notifications on my phone. The prayer time notifications are not showing up at all."},
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "ticket_created" not in st.session_state:
    st.session_state.ticket_created = False
if "ticket_id" not in st.session_state:
    st.session_state.ticket_id = None
if "screenshot" not in st.session_state:
    st.session_state.screenshot = None
if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ☪️ Islam360 Support")
    st.markdown("AI Support Bot — Test Interface")
    st.divider()

    if st.button("✦ New Conversation", type="primary", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.display_messages = []
        st.session_state.ticket_created = False
        st.session_state.ticket_id = None
        st.session_state.screenshot = None
        st.session_state.prefill_query = None
        st.rerun()

    st.divider()
    st.markdown("**QUICK TEST CASES**")

    for tc in TEST_CASES:
        if st.button(f"{tc['id']} — {tc['name']}", key=tc["id"], use_container_width=True):
            st.session_state.prefill_query = tc["query"]
            st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header-bar">
    <span style="font-size:1.8rem">☪️</span>
    <div>
        <h2>Islam360 Customer Support</h2>
        <small>AI Support Bot — Test Interface</small>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Render existing chat messages
# ---------------------------------------------------------------------------
if not st.session_state.display_messages:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #666;">
        <div style="font-size:3rem">💬</div>
        <h3>Assalamualaikum!</h3>
        <p>Type a message below or pick a quick test case from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.display_messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            if msg.get("image"):
                st.image(msg["image"], width=260)
            if msg.get("content"):
                st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Handle prefilled query from sidebar test case click
# ---------------------------------------------------------------------------
if st.session_state.prefill_query:
    user_input = st.session_state.prefill_query
    st.session_state.prefill_query = None

    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, _, st.session_state.chat_history = ask(
                user_input, st.session_state.chat_history
            )
        st.markdown(answer)

    st.session_state.display_messages.append({"role": "agent", "content": answer})
    st.rerun()

# ---------------------------------------------------------------------------
# Chat input
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

    if not user_input.strip() and not attached:
        st.stop()

    pending: dict | None = None
    if attached:
        pending = {
            "bytes": attached.getvalue(),
            "name": attached.name,
            "type": attached.type,
        }
        if st.session_state.screenshot is None:
            st.session_state.screenshot = pending

    display_msg: dict = {"role": "user", "content": user_input}
    if pending:
        display_msg["image"] = pending["bytes"]

    st.session_state.display_messages.append(display_msg)

    with st.chat_message("user"):
        if pending:
            st.image(pending["bytes"], width=260)
        if user_input:
            st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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
