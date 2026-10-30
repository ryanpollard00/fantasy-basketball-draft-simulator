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

def add_fantasy_points(df):
    """Add season fantasy points using the scoring rules in config.SCORING."""
    df = df.copy()
    df["FANTASY_PTS"] = 0
    for stat, points in config.SCORING.items():
        df["FANTASY_PTS"] += df[stat] * points
    return df

def add_per_game(df):
    """Add fantasy points per game, which compares players fairly across missed games and shortened seasons."""
    df = df.copy()
    df["FANTASY_PPG"] = df["FANTASY_PTS"] / df["GP"]
    return df


if __name__ == "__main__":
    df = load_player_seasons()
    df = add_missed_shots(df)
    df = add_fantasy_points(df)
    df = add_per_game(df)
    season = df[(df["SEASON"] == config.TEST_SEASON) & (df["MIN"] >= config.MIN_MINUTES)]
    top = season.nlargest(10, "FANTASY_PPG")
    print(top[["PLAYER_NAME", "GP", "FANTASY_PTS", "FANTASY_PPG"]].round(1).to_string(index=False))