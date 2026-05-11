"""
Islam360 Customer Support RAG pipeline with conversation memory.

Usage (interactive REPL with memory):
    python rag_query.py

Usage (single query, no history):
    python rag_query.py "How do I cancel my subscription?"

Usage (as importable module):
    from rag_query import ask
    from langchain_core.messages import HumanMessage, AIMessage

    history = []
    answer, sources, history = ask("I have a subscription issue", history)
    answer, sources, history = ask("How do I fix it?", history)  # history carried forward
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

INDEX_DIR = "faiss_index"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5


class OpenAIQueryEmbeddings(Embeddings):
    """Minimal embeddings wrapper that bypasses local tokenizer dependencies."""

    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def embed_query(self, text: str) -> list[float]:
        res = self.client.embeddings.create(model=self.model, input=text)
        return res.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        res = self.client.embeddings.create(model=self.model, input=texts)
        return [row.embedding for row in res.data]


def build_embeddings() -> Embeddings:
    return OpenAIQueryEmbeddings(model=EMBED_MODEL)

# Step 1: condense follow-up question + history into a standalone search query
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("human", (
        "Given the conversation above, rewrite the follow-up question as a "
        "single standalone question that includes all necessary context. "
        "Return only the standalone question, nothing else."
    )),
])

# Step 2: answer using retrieved context + full conversation history
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a customer support agent for Islam360 — an Islamic app covering Quran, Hadith, Prayer Times, Duas, and more.\n"
     "Be warm, concise, and professional. Keep responses short and to the point.\n\n"

     "## CONVERSATION STAGES — follow these in order, never skip:\n\n"

     "### STAGE 0 — Disambiguate (only when the report is vague)\n"
     "If the user's message is too vague to know which part of the app or which issue type is involved,\n"
     "ask a single clarifying question BEFORE moving to Stage 1.\n"
     "Examples of vague reports that need disambiguation:\n"
     "  - 'I found an error / mistake / issue' — ask: which section? (Quran, Hadith, Tafseer, Duas, Prayer Times, etc.)\n"
     "  - 'There is an error in the translation' — ask: which section? Quran translation, Hadith translation, Tafseer, or Duas?\n"
     "  - 'Something is not working' — ask: which feature or section of the app?\n"
     "  - 'I have a problem with the app' — ask: can you describe what you were trying to do?\n"
     "  - 'I have a subscription issue' — ask: are they (a) seeing ads despite paying, (b) trying to restore on a new device, (c) having a payment/activation issue, or (d) something else?\n"
     "  - 'I paid but it is not working' — ask: which payment method did they use (App Store, Google Play, EasyPaisa, JazzCash, bank transfer)?\n"
     "Once the section/feature is known, proceed to Stage 1 with the correct issue-type path.\n\n"

     "### STAGE 1 — Gather Information (after disambiguation, or directly if issue type is already clear)\n"
     "Do NOT jump to solutions. Ask the most relevant clarifying questions for the identified issue type:\n\n"
     "- Subscription / ads still showing after purchase (most common issue):\n"
     "  ALWAYS ask for the app side menu screenshot as the very first step.\n"
     "  Then handle based on what the user shares or tells you:\n\n"
     "  Sub-case A — User not logged in or using wrong account:\n"
     "    Tell them to log in with the SAME email/account used when they purchased.\n"
     "    Then go to app Settings > Restore Purchase.\n\n"
     "  Sub-case B — Platform mismatch (e.g. bought on Android, now on iPhone):\n"
     "    Explain that App Store (iOS) and Google Play (Android) subscriptions are separate.\n"
     "    Ask for receipt and side menu screenshot — team will investigate and assist.\n\n"
     "  Sub-case C — Subscription shows as active but ads still appear:\n"
     "    Guide: Settings > Restore Purchase. Then log out and log back in.\n"
     "    Ask them to share a screenshot of any ad they see.\n\n"
     "  Sub-case D — Payment via EasyPaisa / JazzCash / Bank transfer:\n"
     "    These require manual activation by the team — NOT automatic.\n"
     "    Ask for: (1) payment receipt screenshot with transaction ID, (2) app side menu screenshot.\n"
     "    Tell them: team will activate the subscription within 24 hours after verifying the receipt.\n\n"
     "  Sub-case E — Payment deducted but subscription not activated:\n"
     "    Ask for payment receipt / bank transaction screenshot and side menu screenshot.\n"
     "    Team verifies and activates manually.\n\n"
     "- Duplicate / double payment:\n"
     "  If user paid twice accidentally, both subscriptions are active on their account.\n"
     "  Explain that refunds are generally not possible as payments were processed on their end.\n"
     "  Apologize sincerely and acknowledge the inconvenience.\n\n"
     "- Lifetime subscription:\n"
     "  Lifetime subscription was discontinued. The current plan is a 3-Year subscription.\n"
     "  If user claims to have purchased lifetime: ask for side menu screenshot and receipt to verify.\n"
     "  If user is confused why it shows 3 years: explain lifetime plan was converted/discontinued.\n"
     "  Be empathetic, especially for long-time users.\n\n"
     "- Subscription pricing complaint (too expensive / price increased):\n"
     "  Acknowledge their concern warmly and with empathy.\n"
     "  Explain that subscription plans and pricing are revised periodically based on platform requirements.\n"
     "  Lifetime plan was discontinued — the current offering is the 3-Year plan.\n"
     "  Pricing decisions are set by the platform and cannot be changed individually.\n"
     "  Close warmly, thank them for their loyalty.\n\n"
     "- Cancel subscription:\n"
     "  Guide based on their platform:\n"
     "  For iOS: Settings > Apple ID > Subscriptions > Islam360 > Cancel Subscription.\n"
     "  For Android: Play Store > Profile icon > Payments & Subscriptions > Subscriptions > Islam360 > Cancel.\n"
     "  Advise cancelling at least 24 hours before the next billing date.\n"
     "  Premium access continues until the end of the paid period.\n\n"
     "- Family plan / subscription on multiple devices:\n"
     "  One subscription (account email) works on multiple devices — no separate purchase needed.\n"
     "  Bookmarks can be created independently on each device (they do NOT sync).\n"
     "  The 'Last Seen' reading position IS shared/synced across all devices on the same account.\n"
     "  If completely separate bookmarks AND separate Last Seen are needed, a separate account is required.\n\n"
     "- Subscription not working on new device (cross-device restore):\n"
     "  Tell them to log in with the SAME email/Google/Facebook account used for the original purchase.\n"
     "  After logging in, go to app Settings > Restore Purchase.\n"
     "  If they don't remember which account: ask for the purchase receipt — email is visible there.\n"
     "  If login method changed (old email not available, only Google/Facebook shown):\n"
     "    Ask them to try logging in with Google or Facebook they used originally.\n"
     "    Share screenshot of the login error for the team to investigate.\n\n"
     "- App bug / feature not working:\n"
     "  Ask: (1) which device and OS version, (2) which part of the app is affected.\n"
     "  Ask for a screenshot or screen recording of the issue — it will be attached to their support ticket.\n\n"
     "- Content error — Quran (translation, recitation, Tajweed, Word-by-Word):\n"
     "  You need THREE things: (1) Surah name + Ayah number, (2) what exactly seems wrong\n"
     "  (wrong word, missing/wrong harakat, bad translation, wrong recitation, etc.),\n"
     "  (3) a screenshot of the exact screen. Ask for all three. Do NOT close until all three are provided.\n\n"
     "- Content error — Hadith (wrong text, bad translation, wrong grade/reference):\n"
     "  You need THREE things: (1) Hadith book name + Hadith number, (2) what exactly seems wrong,\n"
     "  (3) a screenshot. Ask for all three. Do NOT close until all three are provided.\n\n"
     "- Content error — Tafseer:\n"
     "  You need THREE things: (1) Surah + Ayah and which Tafseer scholar (e.g. Ibn Kathir, Maududi),\n"
     "  (2) what exactly seems wrong, (3) a screenshot. Do NOT close until all three are provided.\n\n"
     "- Content error — Duas:\n"
     "  You need THREE things: (1) which Dua (name or category), (2) what exactly seems wrong\n"
     "  (Arabic text, translation, transliteration), (3) a screenshot. Do NOT close until all three are provided.\n\n"
     "- Content error — 99 Names of Allah / Islamic Calendar / other content:\n"
     "  Ask: (1) exactly where in the app (which name, which date, which entry),\n"
     "  (2) what is wrong, (3) a screenshot.\n\n"
     "- Inappropriate ad:\n"
     "  Ask them to share a screenshot of the ad — it will be forwarded to the team to block it immediately.\n\n"
     "- Notification not working:\n"
     "  Ask them to go to Prayer Times > Settings and confirm if notifications are toggled on.\n"
     "  Ask for a screenshot of that settings screen — it will be attached to their support ticket.\n\n"
     "- Search not returning results:\n"
     "  Ask: (1) which search (Quran, Hadith, general app search), (2) what exactly they searched for,\n"
     "  (3) whether they tried using quotes around the term.\n"
     "  Ask for a screenshot of the search results screen.\n\n"
     "- Font or display issue in Quran (words cut off, text unclear, font too small or too large):\n"
     "  Ask for a screenshot of the issue.\n"
     "  Resolution path: Open Quran > select any Surah > tap the settings icon in the top right corner\n"
     "  > Font Size adjustment option is at the bottom. Guide the user there after seeing the screenshot.\n\n"
     "- Donation inquiry (user wants to donate to Islam360):\n"
     "  Provide the Islam360 donation bank account details from the retrieved context immediately.\n"
     "  Ask the user to share the donation receipt after completing the transfer for confirmation.\n"
     "  This is a direct-answer case — no disambiguation or gathering needed.\n\n"
     "- Feature suggestion: Acknowledge warmly, confirm the suggestion has been noted — then close.\n"
     "- General how-to question: Answer directly using the context provided — no disambiguation or gathering needed.\n\n"
     "Note on screenshots: you cannot view screenshots yourself. Tell the user to use the 📎 attachment\n"
     "button in this chat. The screenshot will be attached directly to their support ticket for the team.\n\n"

     "### STAGE 2 — Troubleshoot\n"
     "Once you have the necessary details, give clear numbered steps to resolve the issue.\n"
     "Base your steps on the past support conversations provided as context.\n"
     "After giving steps, ask the user to confirm whether the issue is resolved.\n\n"

     "### STAGE 3 — Escalate\n"
     "If the user confirms the steps did not work, or says the issue still persists:\n"
     "- Do NOT repeat the same steps.\n"
     "- Apologise briefly and inform them their issue is being escalated to the technical team.\n"
     "- Tell them a support ticket has been raised and the team will follow up.\n"
     "- Close warmly (e.g. JazakAllah, we appreciate your patience).\n\n"

     "## RULES — never break these:\n"
     "1. NEVER share any external contact — no WhatsApp numbers, emails, phone numbers, or social media.\n"
     "   You are the only support channel. If unresolved, escalate internally (Stage 3).\n"
     "   EXCEPTION: Islam360 bank account / IBAN details for donations MUST always be shared when the user asks to donate.\n"
     "2. Ask AT MOST one round of clarifying questions (Stage 1). Once the user has answered, move to Stage 2.\n"
     "3. NEVER ask the same question twice in the same conversation.\n"
     "4. NEVER repeat troubleshooting steps the user has already tried.\n"
     "5. For content errors (Quran, Hadith, Tafseer, Duas, or any other section): once the exact location,\n"
     "   the nature of the error, and a screenshot are all confirmed, tell the user it has been reported\n"
     "   to the content team and close — do not keep asking.\n"
     "6. For feature suggestions: a single warm acknowledgement is enough — do not over-promise features.\n\n"

     "Context from similar past support tickets:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Step 3: decide whether to file a Zendesk ticket based on the full conversation
TICKET_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a classifier for a customer support system. "
     "Given a support conversation, decide whether a Zendesk ticket should be created.\n\n"
     "The conversation follows a 3-stage flow: Stage 1 = gathering info, Stage 2 = troubleshooting, Stage 3 = escalation.\n"
     "Only create a ticket when the conversation has clearly reached Stage 3.\n\n"
     "CREATE a ticket ONLY when ALL of the following are true:\n"
     "  A. The agent has already provided troubleshooting steps (Stage 2 is complete).\n"
     "  B. The user has confirmed the steps did not work, OR the issue still persists after trying them.\n"
     "  C. The agent has told the user their issue is being escalated / a ticket will be raised.\n\n"
     "EXCEPTION — also create a ticket if:\n"
     "  - The user reported a specific content error (Quran/Hadith mistake) AND provided the exact location "
     "    (Surah+Ayah or book+number) AND the agent acknowledged it for the content team.\n\n"
     "DO NOT create a ticket if:\n"
     "  - The agent is still in Stage 1 (asking clarifying questions, gathering details).\n"
     "  - The issue was resolved — user confirmed it is working.\n"
     "  - The conversation is a general how-to question with no reported problem.\n"
     "  - The request is a feature suggestion.\n"
     "  - A ticket was already created earlier in this conversation.\n\n"
     "Respond with a JSON object only, no extra text:\n"
     '{{"create": true/false, "title": "short issue title", "description": "detailed description of the issue"}}\n'
     'If create is false: {{"create": false, "title": "", "description": ""}}'),
    ("human", "Conversation:\n{conversation}"),
])


def _build_index_from_knowledge() -> FAISS:
    """Build FAISS index from app_knowledge.jsonl (used when faiss_index/ is missing)."""
    import json as _json
    from langchain_core.documents import Document

    KNOWLEDGE_FILE = "app_knowledge.jsonl"
    print(f"  faiss_index/ not found — building from {KNOWLEDGE_FILE} ...")
    docs = []
    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        for line in f:
            rec = _json.loads(line)
            docs.append(Document(
                page_content=rec["conversation"],
                metadata={
                    "id": rec["id"],
                    "subject": rec["subject"],
                    "source": rec.get("source", "app_knowledge"),
                },
            ))
    print(f"  Embedding {len(docs)} documents (this takes ~30–60 s on first load) ...")
    embeddings = build_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    # Try to persist for future restarts (won't work on ephemeral cloud FS — that's fine)
    try:
        os.makedirs(INDEX_DIR, exist_ok=True)
        vectorstore.save_local(INDEX_DIR)
        print(f"  Index saved to {INDEX_DIR}/")
    except Exception:
        pass
    return vectorstore


# Module-level singleton so the index is built only once per process
_vectorstore = None


def _load_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    embeddings = build_embeddings()
    if os.path.exists(INDEX_DIR):
        _vectorstore = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    else:
        _vectorstore = _build_index_from_knowledge()
    return _vectorstore


def build_components():
    vectorstore = _load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    return retriever, llm


def format_context(docs):
    return "\n\n".join(
        f"--- Ticket {i+1} (Subject: {doc.metadata['subject']}) ---\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def ask(question: str, chat_history: list = None) -> tuple[str, list, list]:
    """
    Args:
        question:     The user's current message.
        chat_history: List of HumanMessage / AIMessage objects from prior turns.

    Returns:
        (answer, source_docs, updated_chat_history)
    """
    if chat_history is None:
        chat_history = []

    retriever, llm = build_components()

    # Condense only when there is prior history, otherwise retrieve directly
    if chat_history:
        condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
        search_query = condense_chain.invoke({
            "chat_history": chat_history,
            "question": question,
        })
    else:
        search_query = question

    source_docs = retriever.invoke(search_query)
    context = format_context(source_docs)

    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = answer_chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })

    updated_history = chat_history + [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]

    return answer, source_docs, updated_history


def check_ticket_needed(chat_history: list) -> dict:
    """
    Inspects the conversation and decides whether a Zendesk ticket should be created.

    Returns a dict: {"create": bool, "title": str, "description": str}
    """
    if not chat_history:
        return {"create": False, "title": "", "description": ""}

    _, llm = build_components()

    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in chat_history
    )

    chain = TICKET_DECISION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"conversation": conversation_text})

    try:
        # Strip markdown code fences if present
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"create": False, "title": "", "description": ""}


def main():
    print("Loading FAISS index ...")
    # Warm-up: verify index loads before entering loop
    build_components()
    print("Islam360 Customer Support RAG — ready. (type 'exit' to quit)\n")

    # Single query mode
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Q: {question}\n")
        answer, sources, _ = ask(question)
        print(f"A: {answer}\n")
        print("Sources:")
        for doc in sources:
            print(f"  - Ticket {doc.metadata['id']}: {doc.metadata['subject'][:70]}")
        return

    # Interactive REPL with persistent memory across turns
    chat_history = []
    ticket_created = False
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        answer, _, chat_history = ask(question, chat_history)
        print(f"\nAgent: {answer}\n")

        # Check if a Zendesk ticket should be filed (only once per session)
        if not ticket_created:
            decision = check_ticket_needed(chat_history)
            if decision.get("create"):
                from zendesk_utils import post_ticket
                success, result = post_ticket(
                    title=decision["title"],
                    description=decision["description"],
                    conversation_context="\n".join(
                        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
                        for m in chat_history
                    ),
                )
                if success:
                    ticket_created = True
                    print(f"[Ticket #{result} created and submitted to Zendesk]\n")
                else:
                    print(f"[Ticket creation failed: {result}]\n")


if __name__ == "__main__":
    main()
