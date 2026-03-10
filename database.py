import sqlite3
from datetime import datetime

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "experiments.db")


def init_db():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER,
    filters INTEGER,
    num_layers INTEGER DEFAULT 2,
    learning_rate REAL,
    optimizer TEXT,
    accuracy REAL,
    hypothesis TEXT,
    timestamp TEXT
)
""")

    # Migrate older schema: add num_layers if missing
    try:
        cursor.execute("ALTER TABLE experiments ADD COLUMN num_layers INTEGER DEFAULT 2")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    conn.close()


def save_experiment(config, accuracy, hypothesis, cycle=0):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO experiments
    (cycle, filters, num_layers, learning_rate, optimizer, accuracy, hypothesis, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        cycle,
        config["filters"],
        config.get("num_layers", 2),
        config["lr"],
        config["optimizer"],
        accuracy,
        hypothesis,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()
    
def experiment_exists(config):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 1 FROM experiments
    WHERE filters=? AND learning_rate=? AND num_layers=? AND optimizer=?
    LIMIT 1
    """, (
        config["filters"],
        config["lr"],
        config["num_layers"],
        config["optimizer"]
    ))

    result = cursor.fetchone()

    conn.close()

    return result is not None