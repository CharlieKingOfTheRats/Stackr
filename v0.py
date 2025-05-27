# Copyright © 2025 ParthenonAI Solutions. All rights reserved.

import os
import openai
import tiktoken
import requests
from bs4 import BeautifulSoup

# Azure OpenAI config
AZURE_OPENAI_KEY = "your-azure-openai-key"
AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2023-05-15"
AZURE_DEPLOYMENT_NAME_GPT4O = "gpt-4o"
AZURE_DEPLOYMENT_NAME_GPT35 = "gpt-35-turbo"

openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION

# Utils
def estimate_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def extract_subject(question, model=AZURE_DEPLOYMENT_NAME_GPT35):
    prompt = [
        {"role": "system", "content": "Summarize this into 3-5 words."},
        {"role": "user", "content": f"'{question}'"}
    ]
    response = openai.ChatCompletion.create(
        engine=model,
        messages=prompt,
        temperature=0,
        max_tokens=30
    )
    return response.choices[0].message["content"].strip()

def check_response_reasoning(question, response_json, model=AZURE_DEPLOYMENT_NAME_GPT35):
    messages = [
        {"role": "system", "content": "You are a financial reviewer. Check if the plan matches the user's goal logically."},
        {"role": "user", "content": f"Question: {question}\nResponse: {response_json}\nLogical? Explain briefly."}
    ]
    tokens = sum(estimate_tokens(m["content"], model=model) for m in messages)
    print(f"[INFO] Token usage for review: {tokens} tokens (~${tokens * 5 / 1_000_000:.4f})")
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0.4,
        max_tokens=400
    )
    return response.choices[0].message["content"]

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

def auto_tool_orchestrator(user_goal):
    print(f"\n[INFO] Starting automated orchestration for user goal: {user_goal}")

    subject = extract_subject(user_goal)
    print(f"[INFO] Extracted subject: {subject}")

    keywords_trigger_search = ['best', 'latest', 'new', 'today', 'top', 'compare', 'current', 'update']
    if any(word in subject.lower() for word in keywords_trigger_search):
        urls = web_search(user_goal)
        print(f"[INFO] Retrieved URLs from search: {urls}")

        aggregated_text = ""
        for url in urls:
            page_text = get_website_text(url)
            if "[ERROR]" not in page_text:
                aggregated_text += page_text[:2000] + "\n\n"
            else:
                print(page_text)
    else:
        print("[INFO] No web search needed, using empty context.")
        aggregated_text = ""

    model = AZURE_DEPLOYMENT_NAME_GPT4O if "complex" in subject.lower() else AZURE_DEPLOYMENT_NAME_GPT35

    role_prompt = (
        "################################################################################\n"
        "# SYSTEM PROMPT – v1.0\n"
        "# Purpose: Guide the assistant to produce helpful, up-to-date, well-sourced,\n"
        "# and user-tailored answers while using the host platform’s tools responsibly.\n"
        "################################################################################\n\n"
        f"You are {model}, an AI assistant.\n\n"
        "──────────────────────────────────────────────────────────────────────────────\n"
        "1. Conversational Style & Tone\n"
        "• Mirror the user’s formality and enthusiasm.\n"
        "• Be natural and responsive.\n"
        "• Keep it brief unless the user signals otherwise.\n\n"
        "2. Dynamic Knowledge\n"
        "• Use web data if required.\n"
        "• Always cite sources.\n\n"
        "3. Tool Usage\n"
        "• Use Python tools for math.\n"
        "• Search the web if freshness matters.\n\n"
        "4. Multi-Step Tasks\n"
        "• Don’t stop midway—handle full task unless unclear.\n\n"
        "5. Citations & Honesty\n"
        "• All external facts must be traceable.\n\n"
        "6. Safety & Compliance\n"
        "• Reject any disallowed requests.\n\n"
        "7. Response Clarity\n"
        "• Use structured, polished output.\n"
        "• Prompt follow-ups.\n"
        "################################################################################\n\n"
        f"User Goal: {user_goal}\n\n"
        f"Context:\n{aggregated_text[:5000]}\n\n"
        "Based on the above, recommend a compact JSON credit card plan with keys:\n"
        "- card_plan\n"
        "- spending_strategy\n"
        "- redemption_plan\n"
        "Cite sources if applicable.\n"
    )

    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": user_goal}
    ]

    total_tokens = sum(estimate_tokens(m["content"], model=model) for m in messages)
    print(f"[INFO] Estimated tokens for generation: {total_tokens}")

    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0.3,
        max_tokens=700
    )
    plan_json = response.choices[0].message["content"]

    review_notes = check_response_reasoning(user_goal, plan_json)

    return {
        "plan_json": plan_json,
        "review_notes": review_notes
    }

if __name__ == "__main__":
    print("🔁 Automated Credit Card Advisor (type 'quit' to exit)\n")
    while True:
        user_goal = input("Enter your credit card goal: ")
        if user_goal.lower() == "quit":
            break

        result = auto_tool_orchestrator(user_goal)

        print("\n🧠 Generated Plan:\n")
        print(result["plan_json"])
        print("\n🧐 Review Notes:\n")
        print(result["review_notes"])
        print("\n---\n")