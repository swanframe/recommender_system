from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


def _ensure_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}."
        )


def _coerce_string(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    return df


def _fill_string(df: pd.DataFrame, cols: list[str], value: str = "unknown") -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(value)
    return df


@dataclass(frozen=True)
class DataPaths:
    users_csv: Path
    items_csv: Path
    events_csv: Path

    @staticmethod
    def from_raw_dir(raw_dir: Path) -> "DataPaths":
        return DataPaths(
            users_csv=raw_dir / "users.csv",
            items_csv=raw_dir / "items.csv",
            events_csv=raw_dir / "events.csv",
        )


class DataLoader:
    """
    Loads and cleans:
      - users.csv:  (user_id, name, age, gender, region)
      - items.csv:  (item_id, title, content_type, genre)
      - events.csv: (user_id, item_id, event_type, watch_seconds, timestamp)
    """

    def __init__(self, paths: DataPaths, encoding: str = "utf-8"):
        self.paths = paths
        self.encoding = encoding

    def load_users(self) -> pd.DataFrame:
        df = pd.read_csv(self.paths.users_csv, encoding=self.encoding)

        # NOTE: disesuaikan dengan contoh kamu: ada kolom "name"
        required = ["user_id", "name", "age", "gender", "region"]
        _ensure_columns(df, required, "users.csv")

        df = _coerce_string(df, ["user_id", "name", "gender", "region"])
        df = _fill_string(df, ["name", "gender", "region"], value="unknown")

        # age -> numeric, missing -> median (fallback 0)
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        if df["age"].notna().any():
            df["age"] = df["age"].fillna(df["age"].median())
        else:
            df["age"] = df["age"].fillna(0)
        df["age"] = df["age"].round().astype("int64")

        # Final dtypes
        df["user_id"] = df["user_id"].astype("string")
        df["name"] = df["name"].astype("string")
        df["gender"] = df["gender"].astype("string")
        df["region"] = df["region"].astype("string")

        return df

    def load_items(self) -> pd.DataFrame:
        df = pd.read_csv(self.paths.items_csv, encoding=self.encoding)

        required = ["item_id", "title", "content_type", "genre"]
        _ensure_columns(df, required, "items.csv")

        df = _coerce_string(df, ["item_id", "title", "content_type", "genre"])
        df = _fill_string(df, ["title", "content_type", "genre"], value="unknown")

        df["item_id"] = df["item_id"].astype("string")
        df["title"] = df["title"].astype("string")
        df["content_type"] = df["content_type"].astype("string")
        df["genre"] = df["genre"].astype("string")

        return df

    def load_events(self) -> pd.DataFrame:
        df = pd.read_csv(self.paths.events_csv, encoding=self.encoding)

        required = ["user_id", "item_id", "event_type", "watch_seconds", "timestamp"]
        _ensure_columns(df, required, "events.csv")

        df = _coerce_string(df, ["user_id", "item_id", "event_type"])
        df = _fill_string(df, ["event_type"], value="unknown")

        # watch_seconds -> numeric, missing -> 0, negative -> 0
        df["watch_seconds"] = pd.to_numeric(df["watch_seconds"], errors="coerce").fillna(0)
        df["watch_seconds"] = df["watch_seconds"].clip(lower=0).round().astype("int64")

        # timestamp -> datetime; invalid jadi NaT
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Final dtypes
        df["user_id"] = df["user_id"].astype("string")
        df["item_id"] = df["item_id"].astype("string")
        df["event_type"] = df["event_type"].astype("string")

        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        users = self.load_users()
        items = self.load_items()
        events = self.load_events()
        return users, items, events


if __name__ == "__main__":
    # Run:
    # python -m recommender_system.data.data_loader
    from recommender_system.config import RAW_DIR

    paths = DataPaths.from_raw_dir(RAW_DIR)
    loader = DataLoader(paths)

    users_df, items_df, events_df = loader.load_all()

    print("USERS:", users_df.shape)
    print(users_df.head(5), "\n")

    print("ITEMS:", items_df.shape)
    print(items_df.head(5), "\n")

    print("EVENTS:", events_df.shape)
    print(events_df.head(5), "\n")

    print("EVENTS dtypes:\n", events_df.dtypes)