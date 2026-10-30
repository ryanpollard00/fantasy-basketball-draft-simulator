"""
Calculate fantasy points and fantasy position eligibility for each player season.

Run from the project root to print a preview:
    python -m src.scoring
"""

import sqlite3

import pandas as pd

import config


def load_player_seasons():
    """Load every player's season totals from the database."""
    conn = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql("SELECT * FROM player_seasons", conn)
    conn.close()
    return df


if __name__ == "__main__":
    df = load_player_seasons()
    print(df.shape)
    print(df.head())