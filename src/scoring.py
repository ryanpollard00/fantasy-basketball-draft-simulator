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

def add_missed_shots(df):
    """Add missed field goals and free throws, which are scored but not provided by the API."""
    df = df.copy()
    df["FG_MISS"] = df["FGA"] - df["FGM"]
    df["FT_MISS"] = df["FTA"] - df["FTM"]
    return df


if __name__ == "__main__":
    df = load_player_seasons()
    df = add_missed_shots(df)
    print(df[["PLAYER_NAME", "SEASON", "FGM", "FGA", "FG_MISS", "FTM", "FTA", "FT_MISS"]].head())
    print("Negative misses:", (df[["FG_MISS", "FT_MISS"]] < 0).sum().sum())