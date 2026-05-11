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
# CSS — match the test_ui.html look exactly
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #f0f2f5 !important;
}

/* Hide Streamlit top bar decoration */
[data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] > div:first-child {
    background-color: #1B5E20 !important;
    padding: 0 !important;
}
[data-testid="stSidebarContent"] {
    background-color: #1B5E20 !important;
    padding: 16px 12px !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: white !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2E7D32 !important;
    margin: 10px 0 !important;
}

/* New Conversation button */
[data-testid="stSidebar"] .stButton > button {
    background-color: #2E7D32 !important;
    color: white !important;
    border: 1px solid #4CAF50 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 6px 10px !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 4px !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #388E3C !important;
    border-color: #81C784 !important;
}
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background-color: #4CAF50 !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    margin-bottom: 2px !important;
}

/* Section label */
[data-testid="stSidebar"] .stMarkdown small {
    color: #A5D6A7 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Main area ── */
[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: white !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 10px !important;
    margin: 6px 16px !important;
    padding: 10px 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #DCF8C6 !important;
    border-color: #b7e4a0 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-color: #4CAF50 !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #2E7D32 !important;
    box-shadow: 0 0 0 2px rgba(46,125,50,0.15) !important;
}

/* Send button inside chat input */
[data-testid="stChatInputSubmitButton"] > button {
    background-color: #2E7D32 !important;
    border-radius: 50% !important;
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

BADGE_COLORS = {
    "TC-01": "#9C27B0", "TC-02": "#1565C0", "TC-03": "#B71C1C",
    "TC-04": "#E65100", "TC-05": "#00695C", "TC-06": "#4527A0",
    "TC-07": "#558B2F", "TC-08": "#F57F17", "TC-09": "#0277BD",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "chat_history": [],
    "display_messages": [],
    "ticket_created": False,
    "ticket_id": None,
    "screenshot": None,
    "prefill_query": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ☪️ Islam360 Support")
    st.markdown("<small>AI Support Bot — Test Interface</small>", unsafe_allow_html=True)
    st.divider()

    if st.button("✦  New Conversation", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v if not isinstance(v, list) else []
        st.rerun()

    st.divider()
    st.markdown("<small>QUICK TEST CASES</small>", unsafe_allow_html=True)
    st.markdown("")

    for tc in TEST_CASES:
        color = BADGE_COLORS.get(tc["id"], "#555")
        label = (
            f'<span style="background:{color};color:white;padding:1px 6px;'
            f'border-radius:4px;font-size:0.7rem;font-weight:700">{tc["id"]}</span> '
            f'{tc["name"]}'
        )
        if st.button(tc["name"], key=tc["id"], use_container_width=True):
            st.session_state.prefill_query = tc["query"]
            st.rerun()

# ---------------------------------------------------------------------------
# Header bar
# ---------------------------------------------------------------------------
st.markdown("""
<div style="background:#1B5E20;color:white;padding:14px 20px;
            border-radius:0;margin-bottom:8px;display:flex;
            align-items:center;gap:14px;">
  <span style="font-size:2rem">☪️</span>
  <div>
    <div style="font-size:1.2rem;font-weight:700;color:white">Islam360 Customer Support</div>
    <div style="font-size:0.82rem;color:#c8e6c9">AI Support Bot — Test Interface</div>
  </div>
  <div style="margin-left:auto;font-size:0.8rem;color:#a5d6a7">
    ● Connected
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------------------
if not st.session_state.display_messages:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;color:#888;">
      <div style="font-size:3rem">💬</div>
      <h3 style="color:#555">Assalamualaikum!</h3>
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
# Handle prefilled query from sidebar button click
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
