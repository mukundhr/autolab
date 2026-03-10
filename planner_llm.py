import sqlite3
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from database import DB_NAME

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


def get_experiment_summary():

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query("SELECT * FROM experiments", conn)

    conn.close()

    if df.empty:
        return "No experiments yet."

    summary = df.groupby(
        ["filters", "num_layers", "learning_rate", "optimizer"]
    )["accuracy"].agg(
        mean="mean",
        runs="count"
    ).reset_index()

    return summary.to_string()


def propose_batch():

    history = get_experiment_summary()

    prompt = f"""
Experiment history:
{history}

Step 1: Reflect on patterns in the experiments.

Step 2: Form a hypothesis.

Step 3: Propose 3 experiments to test it.

Constraints:
filters must be one of [16,32,64,128]
learning_rate must be one of [0.01,0.001]
optimizer must be "adam" or "sgd"
num_layers must be one of [1,2,3]

Return ONLY JSON:

{{
 "reflection": "...",
 "hypothesis": "...",
 "experiments": [
   {{"filters": 64, "learning_rate": 0.001, "optimizer": "adam", "num_layers": 2}},
   {{"filters": 128, "learning_rate": 0.001, "optimizer": "adam", "num_layers": 2}},
   {{"filters": 32, "learning_rate": 0.001, "optimizer": "sgd", "num_layers": 1}}
 ]
}}
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-5.3-chat"),
        messages=[
            {"role": "system", "content": "You are an ML research assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content

    # Extract JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)

    if not match:
        raise ValueError("No JSON found in LLM response:\n" + text)

    json_text = match.group(0)

    return json.loads(json_text)