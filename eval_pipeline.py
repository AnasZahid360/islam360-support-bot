"""
Islam360 Customer Support Bot — Evaluation Pipeline
=====================================================
Tests the bot against 9 curated repetitive use cases and generates
a pass/fail quality report with per-case scores.

Usage:
    python eval_pipeline.py              # run all 9 test cases
    python eval_pipeline.py --tc TC-01   # run a single test case by ID
    python eval_pipeline.py --list       # list all available test cases

Output:
    - Prints pass/fail + score for each case to stdout
    - Saves detailed eval_report.json on completion

Requires:
    OPENAI_API_KEY in .env
    faiss_index/ folder (run build_index.py if missing)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_query import ask

load_dotenv()

# ---------------------------------------------------------------------------
# 9 Curated Test Cases (derived from repetitive_questions.csv)
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id": "TC-01",
        "name": "Subscription — Ads Still Showing After Purchase",
        "category": "subscription",
        "query": (
            "I purchased monthly subscription yesterday but still when I open app "
            "to read quran, I get to see ads"
        ),
        "expected_behavior": (
            "Bot must be in Stage 1 (gathering info). "
            "Must ask the user to share a screenshot of the side menu AND confirm they are logged in / "
            "ask about their platform (Android, iOS, or Huawei). "
            "Should NOT give troubleshooting steps (logout/login etc.) before seeing the screenshot."
        ),
    },
    {
        "id": "TC-02",
        "name": "Font / Display Issue — Quran Words Cut in Half",
        "category": "app_display",
        "query": (
            "During the Quran reading some words are not clear on the screen, "
            "some words are cut in half."
        ),
        "expected_behavior": (
            "Bot should ask the user to share a screenshot of the issue. "
            "Should mention or provide the font size fix path: "
            "Quran > any Surah > settings icon (top right) > Font Size at the bottom. "
            "Should NOT skip the screenshot step before guiding."
        ),
    },
    {
        "id": "TC-03",
        "name": "Inappropriate Ad Reported",
        "category": "inappropriate_ad",
        "query": (
            "Please remove this ad. It shows things that are hurtful and inappropriate, "
            "especially in front of the Quran."
        ),
        "expected_behavior": (
            "Bot must apologize sincerely for the inappropriate content. "
            "Must ask the user to share a screenshot of the ad so the team can block it immediately. "
            "Tone should be empathetic and urgent. "
            "Should NOT dismiss the complaint or give a generic reply without asking for the screenshot."
        ),
    },
    {
        "id": "TC-04",
        "name": "Quran Content Mistake Reported",
        "category": "content_error_quran",
        "query": (
            "Assalamualaikum. There is a mistake in the Quran in the app. "
            "There is a missing zabar on alif which changes the pronunciation completely."
        ),
        "expected_behavior": (
            "Bot must be in Stage 1 and collect THREE things before closing: "
            "(1) Surah name and Ayah number, "
            "(2) what exactly seems wrong (missing harakat, wrong word, etc.), "
            "(3) a screenshot of the exact screen. "
            "Must NOT jump to Stage 2 without all three pieces of information. "
            "Should NOT immediately say 'we checked and it is correct' without collecting the details first."
        ),
    },
    {
        "id": "TC-05",
        "name": "Web / Desktop App Feature Request",
        "category": "feature_suggestion",
        "query": (
            "Assalamualaikum Islam360 Staff. Do you have a desktop or web application? "
            "I spend 8-9 hours at work on a laptop and would love to use Islam360 there."
        ),
        "expected_behavior": (
            "Bot should acknowledge the request warmly and in an Islamic tone. "
            "Must inform that Islam360 is currently a mobile app with no dedicated web/desktop version. "
            "Should suggest using an Android emulator like BlueStacks as a workaround. "
            "Should confirm the suggestion has been noted. "
            "Should NOT promise a web app will be built."
        ),
    },
    {
        "id": "TC-06",
        "name": "Arabic Hadith Search Not Returning Results",
        "category": "search_issue",
        "query": (
            "I have an issue with search. I want to search for any Arabic word or sentence "
            "in Hadith books but it never gives results even when the word is definitely in the book."
        ),
        "expected_behavior": (
            "Bot should ask what search term the user tried and in which Hadith book. "
            "Must suggest placing the search term inside inverted commas / quotation marks for better results. "
            "May ask for a screenshot or screen recording of the search attempt. "
            "Should NOT just ask for a screen recording without also offering the inverted commas tip."
        ),
    },
    {
        "id": "TC-07",
        "name": "Feature Suggestion — Add Shia Books",
        "category": "feature_suggestion",
        "query": "Assalamualaikum. Can you please add Shia books to the Islam360 app as well?",
        "expected_behavior": (
            "Bot should acknowledge the suggestion warmly with an Islamic greeting. "
            "Must confirm the suggestion has been noted and will be shared with the team. "
            "Should NOT over-promise or commit to adding the content. "
            "Should close the conversation warmly and concisely."
        ),
    },
    {
        "id": "TC-08",
        "name": "Donation Account / IBAN Request",
        "category": "donation",
        "query": (
            "Assalam o alaikum. I want to donate to Islam360. "
            "Can you give me the account number and IBAN for overseas donations?"
        ),
        "expected_behavior": (
            "Bot must provide Islam360's official bank account details including: "
            "Bank Alfalah Ltd, Title: International Technology Mission, "
            "IBAN starting with PK65ALFH... and/or PK71ALFH..., Branch Code 5592, Karachi. "
            "Should ask the user to share the donation receipt after transferring. "
            "Should be warm and appreciative of the donation intent. "
            "Must NOT refuse to share the bank details citing the no-contact-sharing rule."
        ),
    },
    {
        "id": "TC-09",
        "name": "Namaz / Prayer Notification Not Working",
        "category": "notification",
        "query": (
            "Salam. I am unable to get Namaz notifications on my phone. "
            "The prayer time notifications are not showing up at all."
        ),
        "expected_behavior": (
            "Bot must direct the user to: Prayer Times section > Settings. "
            "Must ask the user to check if the notification toggle is turned on. "
            "Must ask for a screenshot of the Prayer Times settings screen. "
            "May also ask for the side menu screenshot to confirm location / account. "
            "Should NOT just ask for a screenshot without explaining where to navigate."
        ),
    },
]

# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict quality evaluator for an Islamic app customer support chatbot.\n"
     "Your job: decide whether the bot's FIRST response correctly handles the user query "
     "based on the expected behavior criteria.\n\n"
     "Evaluate on ALL of these dimensions:\n"
     "1. Correct Stage — Stage 1 = ask clarifying questions / gather info; "
     "Stage 2 = provide troubleshooting steps; "
     "direct-answer cases (suggestions, how-to, donation) = answer immediately.\n"
     "2. Required Content — does the response include all required questions or information specified in expected behavior?\n"
     "3. Tone — warm, professional, Islamic greeting (WaAlaikumAssalam / Salam), concise.\n"
     "4. No rule violations — does not share disallowed contacts; does not repeat; "
     "does not skip info-gathering when it is required.\n\n"
     "Be strict: if any required content item from the expected behavior is missing, score lower and mark as FAIL.\n\n"
     "Respond ONLY with a valid JSON object — no markdown, no extra text:\n"
     '{{"pass": true/false, "score": 0-10, '
     '"reason": "one concise sentence summarising the verdict", '
     '"missing": "comma-separated list of what was missing or wrong (empty string if fully passed)"}}'),
    ("human",
     "Test Case: [{test_id}] {test_name}\n\n"
     "User Query:\n{query}\n\n"
     "Expected Behavior:\n{expected}\n\n"
     "Bot's Response:\n{response}"),
])


def run_judge(llm: ChatOpenAI, tc: dict, response: str) -> dict:
    """Runs the LLM judge on a single test case and returns a judgment dict."""
    chain = JUDGE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "test_id": tc["id"],
        "test_name": tc["name"],
        "query": tc["query"],
        "expected": tc["expected_behavior"],
        "response": response,
    })
    # Strip markdown fences if present
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "pass": False,
            "score": 0,
            "reason": "Judge returned unparseable output",
            "missing": raw[:200],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_case(tc: dict, judge_llm: ChatOpenAI) -> dict:
    """Runs a single test case end-to-end and returns a result dict."""
    print(f"\n[{tc['id']}] {tc['name']}")
    print(f"  Category : {tc['category']}")
    print(f"  Query    : {tc['query'][:100]}{'...' if len(tc['query']) > 100 else ''}")

    # Call the bot (single turn, no prior history)
    try:
        bot_response, _, _ = ask(tc["query"], [])
    except Exception as e:
        bot_response = f"BOT_ERROR: {e}"
        judgment = {
            "pass": False,
            "score": 0,
            "reason": f"Bot threw an exception: {e}",
            "missing": "",
        }
        print(f"  Result   : ❌ ERROR — {e}")
        return {**tc, "bot_response": bot_response, "judgment": judgment}

    print(f"  Bot      : {bot_response[:120]}{'...' if len(bot_response) > 120 else ''}")

    # Judge
    judgment = run_judge(judge_llm, tc, bot_response)
    verdict = "✅ PASS" if judgment.get("pass") else "❌ FAIL"
    print(f"  Result   : {verdict}  |  Score: {judgment.get('score', 0)}/10")
    print(f"  Reason   : {judgment.get('reason', '')}")
    if judgment.get("missing"):
        print(f"  Missing  : {judgment['missing']}")

    return {**tc, "bot_response": bot_response, "judgment": judgment}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def save_report(results: list[dict], report_path: str = "eval_report.json") -> None:
    passed = sum(1 for r in results if r["judgment"].get("pass"))
    total = len(results)
    avg_score = sum(r["judgment"].get("score", 0) for r in results) / total if total else 0

    report = {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{100 * passed // total}%" if total else "0%",
            "avg_score": round(avg_score, 1),
        },
        "results": [
            {
                "id": r["id"],
                "name": r["name"],
                "category": r["category"],
                "query": r["query"],
                "bot_response": r["bot_response"],
                "judgment": r["judgment"],
            }
            for r in results
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report["summary"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Islam360 Support Bot — Evaluation Pipeline"
    )
    parser.add_argument(
        "--tc",
        type=str,
        default=None,
        metavar="ID",
        help="Run a specific test case (e.g. TC-01). Omit to run all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available test case IDs and names, then exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable test cases:")
        for tc in TEST_CASES:
            print(f"  {tc['id']}  [{tc['category']:25s}]  {tc['name']}")
        return

    # Filter cases
    if args.tc:
        cases = [tc for tc in TEST_CASES if tc["id"].upper() == args.tc.upper()]
        if not cases:
            print(f"Error: Test case '{args.tc}' not found. Use --list to see all IDs.")
            return
    else:
        cases = TEST_CASES

    print("=" * 65)
    print("  Islam360 Customer Support Bot — Evaluation Pipeline")
    print(f"  Running {len(cases)} test case(s)")
    print("=" * 65)

    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    results = []

    for tc in cases:
        result = run_case(tc, judge_llm)
        results.append(result)

    # Summary (only when running multiple cases)
    if len(results) > 1:
        passed = sum(1 for r in results if r["judgment"].get("pass"))
        total = len(results)
        avg_score = sum(r["judgment"].get("score", 0) for r in results) / total

        print("\n" + "=" * 65)
        print(f"  FINAL RESULTS : {passed}/{total} passed  ({100 * passed // total}%)")
        print(f"  Average Score : {avg_score:.1f} / 10")
        print("=" * 65)

        failures = [r for r in results if not r["judgment"].get("pass")]
        if failures:
            print("\n⚠️  Failed cases:")
            for r in failures:
                print(f"   [{r['id']}] {r['name']}")
                print(f"          → {r['judgment'].get('reason', '')}")
        else:
            print("\n🎉 All test cases passed!")

    # Save report
    report_path = "eval_report.json"
    summary = save_report(results, report_path)
    print(f"\n📄 Detailed report saved → {report_path}")
    if len(results) == 1:
        r = results[0]
        verdict = "✅ PASSED" if r["judgment"].get("pass") else "❌ FAILED"
        print(f"   {verdict}  |  Score: {r['judgment'].get('score', 0)}/10")


if __name__ == "__main__":
    main()
