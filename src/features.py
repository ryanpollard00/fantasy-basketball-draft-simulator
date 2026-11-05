"""
Build the model's features: one row per player per target season, with
inputs from earlier seasons and the target season's actual fantasy points.

Run from the project root to print a preview:
    python -m src.features
"""

import sqlite3

import pandas as pd

import config
from src.scoring import (
    add_fantasy_points,
    add_missed_shots,
    add_per_game,
    add_positions,
    load_player_seasons,
    load_players,
)


def load_team_seasons():
    """Load each team's games played and win % for every season."""
    conn = sqlite3.connect(config.DB_PATH)
    teams = pd.read_sql(
        "SELECT SEASON, TEAM_ID, WINS + LOSSES AS TEAM_GAMES, WIN_PCT FROM team_seasons",
        conn,
    )
    conn.close()
    return teams


def add_rate_stats(df, teams):
    """Add team context plus the per-game and share stats that stay comparable in shortened seasons."""
    df = df.merge(teams, on=["SEASON", "TEAM_ID"], how="left", validate="many_to_one")
    df["GP_SHARE"] = (df["GP"] / df["TEAM_GAMES"]).clip(upper=1)
    df["MPG"] = df["MIN"] / df["GP"]
    return df


def build_base_table():
    """One row per player season with fantasy points, positions and rate stats."""
    df = load_player_seasons()
    df = add_missed_shots(df)
    df = add_fantasy_points(df)
    df = add_per_game(df)
    df = add_positions(df, load_players())
    df = add_rate_stats(df, load_team_seasons())
    return df


if __name__ == "__main__":
    base = build_base_table()
    print(base.shape)
    print("Player seasons with no team match:", base["TEAM_GAMES"].isna().sum())
    print(base[["FANTASY_PPG", "GP_SHARE", "MPG", "WIN_PCT"]].describe().round(2))