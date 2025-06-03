# Copyright © 2025 PantheonAI. All rights reserved.
import os
import openai
import tiktoken
import requests
import sqlite3
from bs4 import BeautifulSoup
from datetime import datetime

# Azure OpenAI Config
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2023-05-15"
AZURE_DEPLOYMENT_NAME_GPT4O = "gpt-4o"
AZURE_DEPLOYMENT_NAME_GPT35 = "gpt-35-turbo"

openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION

# SQLite setup
DB_PATH = "trend_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_goal TEXT,
                    roi_estimate REAL,
                    consistency_score REAL
                )''')
    conn.commit()
    conn.close()

def log_metrics(user_goal, roi_estimate, consistency_score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO analysis_log (timestamp, user_goal, roi_estimate, consistency_score)
                 VALUES (?, ?, ?, ?)''', (datetime.now(), user_goal, roi_estimate, consistency_score))
    conn.commit()
    conn.close()

def estimate_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def extract_subject(question, model=AZURE_DEPLOYMENT_NAME_GPT35):
    messages = [
        {"role": "system", "content": "Summarize this into 3-5 words."},
        {"role": "user", "content": f"'{question}'"}
    ]
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0,
        max_tokens=30
    )
    return response.choices[0].message["content"].strip()

def get_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"[ERROR] Failed to load content from {url}: {e}"

def web_search(query, max_results=3):
    print(f"[INFO] Running web search for query: {query}")
    return [
        "https://www.nerdwallet.com/best-credit-cards",
        "https://www.creditkarma.com/credit-cards/best-credit-cards",
        "https://www.forbes.com/advisor/credit-cards/best-credit-cards/"
    ][:max_results]

def estimate_roi(plan_text, model=AZURE_DEPLOYMENT_NAME_GPT35):
    messages = [
        {"role": "system", "content": "You are an ROI analyst. Estimate annual value of this plan in dollars."},
        {"role": "user", "content": plan_text}
    ]
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0.3,
        max_tokens=100
    )
    text = response.choices[0].message["content"]
    try:
        return float("".join(c for c in text if c.isdigit() or c == '.'))
    except:
        return 0.0

def check_self_consistency(user_goal, model=AZURE_DEPLOYMENT_NAME_GPT35):
    messages = [
        {"role": "system", "content": "Generate a credit card strategy for this user goal."},
        {"role": "user", "content": user_goal}
    ]
    plans = []
    for _ in range(3):
        response = openai.ChatCompletion.create(
            engine=model,
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
        plans.append(response.choices[0].message["content"])

    unique_plans = set(plans)
    consistency_score = 1.0 - (len(unique_plans) - 1) / 2.0  # 1 = same, 0 = all different
    return round(consistency_score, 2)

def check_response_reasoning(question, response_json, model=AZURE_DEPLOYMENT_NAME_GPT35):
    messages = [
        {"role": "system", "content": "You are a financial reviewer. Check if the plan matches the user's goal logically."},
        {"role": "user", "content": f"Question: {question}\nResponse: {response_json}\nLogical? Explain briefly."}
    ]
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0.4,
        max_tokens=400
    )
    return response.choices[0].message["content"]

def auto_tool_orchestrator(user_goal):
    print(f"\n[INFO] Starting auto orchestration for goal: {user_goal}")

    subject = extract_subject(user_goal)
    print(f"[INFO] Extracted subject: {subject}")

    keywords_trigger_search = ['best', 'latest', 'new', 'today', 'top', 'compare', 'current', 'update']
    if any(word in subject.lower() for word in keywords_trigger_search):
        urls = web_search(user_goal)
        aggregated_text = ""
        for url in urls:
            text = get_website_text(url)
            if "[ERROR]" not in text:
                aggregated_text += text[:2000] + "\n\n"
    else:
        aggregated_text = ""

    model = AZURE_DEPLOYMENT_NAME_GPT4O if "complex" in subject.lower() else AZURE_DEPLOYMENT_NAME_GPT35

    role_prompt = (
        "You are Stackr, an expert credit card optimizer.\n\n"
        "1. Tailor your plan to the user goal.\n"
        "2. Use real rewards logic.\n"
        "3. Format output in JSON:\n"
        "  - card_plan\n  - spending_strategy\n  - redemption_plan\n"
        "User Goal: " + user_goal + "\n\nContext:\n" + aggregated_text[:4000]
    )

    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": user_goal}
    ]

    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0.3,
        max_tokens=700
    )
    plan_json = response.choices[0].message["content"]

    # Add ROI Estimator + Consistency Check
    roi = estimate_roi(plan_json)
    consistency_score = check_self_consistency(user_goal)
    review_notes = check_response_reasoning(user_goal, plan_json)

    log_metrics(user_goal, roi, consistency_score)

    return {
        "plan_json": plan_json,
        "roi_estimate": roi,
        "consistency_score": consistency_score,
        "review_notes": review_notes
    }

if __name__ == "__main__":
    init_db()
    print("🧠 Stackr – Credit Card Optimizer")
    print("Type 'quit' to exit.\n")

    while True:
        user_goal = input("Enter your credit card goal: ")
        if user_goal.lower() == "quit":
            break

        result = auto_tool_orchestrator(user_goal)

        print("\n📝 Generated Plan:\n")
        print(result["plan_json"])
        print(f"\n💸 ROI Estimate: ${result['roi_estimate']}")
        print(f"🔁 Consistency Score: {result['consistency_score']}")
        print(f"\n🧐 Review Notes:\n{result['review_notes']}")
        print("\n" + "-"*80 + "\n")