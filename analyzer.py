import sqlite3
import pandas as pd
from database import DB_NAME


def analyze_experiments():

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query("SELECT * FROM experiments", conn)

    conn.close()

    if df.empty:
        print("No experiments yet")
        return

    # ---- Ordered experiment log ----
    print("\nExperiments by Cycle\n")

    ordered = df.sort_values(
        ["cycle", "accuracy"],
        ascending=[True, False]
    )

    print(ordered[
        ["cycle", "filters", "num_layers", "learning_rate", "optimizer", "accuracy"]
    ])

    # ---- Aggregate statistics ----
    grouped = df.groupby(
        ["filters", "num_layers", "learning_rate", "optimizer"]
    )["accuracy"].agg(
        mean_accuracy="mean",
        runs="count"
    ).reset_index()

    grouped = grouped.sort_values("mean_accuracy", ascending=False)

    print("\nExperiment Summary\n")
    print(grouped)

    # ---- Best configuration ----
    best = grouped.iloc[0]

    print("\nBest Configuration:\n")
    print(best)


IMPROVEMENT_THRESHOLD = 0.1


def is_significant_improvement(new_accuracy):

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query("SELECT accuracy FROM experiments", conn)

    conn.close()

    if df.empty:
        return True

    best_accuracy = df["accuracy"].max()

    improvement = new_accuracy - best_accuracy

    return improvement >= IMPROVEMENT_THRESHOLD


if __name__ == "__main__":
    analyze_experiments()