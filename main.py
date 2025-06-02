import os
import json
import openai
import uvicorn
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from azure.cosmos import CosmosClient, PartitionKey

# Load config
COSMOS_URI = os.getenv("COSMOS_URI", "")
COSMOS_KEY = os.getenv("COSMOS_KEY", "")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "creditOptimizerDB")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER", "logs")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_DEPLOYMENT_NAME_GPT4O = os.getenv("AZURE_DEPLOYMENT_NAME_GPT4O", "gpt-4o")

openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION

app = FastAPI(title="Credit Card Optimizer")

# Connect to Cosmos DB if configured
cosmos_client = None
container = None
if COSMOS_URI and COSMOS_KEY:
    cosmos_client = CosmosClient(COSMOS_URI, COSMOS_KEY)
    database = cosmos_client.create_database_if_not_exists(id=COSMOS_DATABASE)
    container = database.create_container_if_not_exists(
        id=COSMOS_CONTAINER,
        partition_key=PartitionKey(path="/user_id"),
        offer_throughput=400
    )

class UserGoal(BaseModel):
    user_id: Optional[str] = "anonymous"
    goal: str

def roi_estimator(plan_json: str) -> float:
    # Placeholder: estimate ROI based on plan contents
    # For demo, just return a dummy score
    return 0.85

def self_consistency_checker(plan_json: str) -> float:
    # Placeholder: compare multiple runs or internal logic for stability
    return 0.9

def log_to_cosmos(user_id: str, roi: float, consistency: float):
    if container:
        container.upsert_item({
            "id": f"{user_id}-{os.urandom(4).hex()}",
            "user_id": user_id,
            "roi_estimate": roi,
            "consistency_score": consistency
        })

@app.post("/optimize")
async def optimize(user_goal: UserGoal):
    if not user_goal.goal:
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")

    prompt = f"User goal: {user_goal.goal}\nGenerate a compact credit card plan with spending and redemption strategies."

    response = openai.ChatCompletion.create(
        engine=AZURE_DEPLOYMENT_NAME_GPT4O,
        messages=[
            {"role": "system", "content": "You are a credit card optimizer assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=700,
    )
    plan_json = response.choices[0].message["content"]

    roi = roi_estimator(plan_json)
    consistency = self_consistency_checker(plan_json)

    # Log metrics
    log_to_cosmos(user_goal.user_id, roi, consistency)

    return {
        "plan": plan_json,
        "roi_estimate": roi,
        "consistency_score": consistency
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)