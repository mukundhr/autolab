import sqlite3
import pandas as pd
import math
from database import DB_NAME


def bandit_score(config):

    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM experiments", conn)
    conn.close()

    if df.empty:
        return 1.0

    subset = df[
        (df["filters"] == config["filters"]) &
        (df["num_layers"] == config.get("num_layers", 2)) &
        (df["learning_rate"] == config["lr"]) &
        (df["optimizer"] == config["optimizer"])
    ]

    if subset.empty:
        return 1.0  # encourage exploration

    mean_acc = subset["accuracy"].mean()
    runs = len(subset)

    total_runs = len(df)

    exploration = math.sqrt(math.log(total_runs + 1) / runs)

    return mean_acc + exploration