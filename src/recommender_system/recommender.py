from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RecommendationResult:
    items: list[dict]
    fallback_used: bool


class Recommender:
    """
    - recommend_popular(k): item terpopuler berdasarkan sum(watch_seconds)
    - recommend_for_user(user_id, k): item-based CF dengan cosine similarity
    - exclude: jangan rekomendasikan item yang sudah ditonton total > 600 detik
    - cold start: fallback ke popular
    """

    def __init__(
        self,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        events_df: pd.DataFrame,
        watch_exclude_threshold: int = 600,
    ):
        self.users_df = users_df.copy()
        self.items_df = items_df.copy()
        self.events_df = events_df.copy()
        self.watch_exclude_threshold = int(watch_exclude_threshold)

        self._popular_df: Optional[pd.DataFrame] = None
        self._user_item: Optional[pd.DataFrame] = None
        self._item_ids: Optional[list[str]] = None
        self._item_index: Optional[dict[str, int]] = None
        self._item_sim: Optional[np.ndarray] = None

        self.fit()

    def fit(self) -> None:
        self._ensure_events_columns()
        self._ensure_items_columns()

        # Normalize types
        self.events_df["user_id"] = self.events_df["user_id"].astype("string")
        self.events_df["item_id"] = self.events_df["item_id"].astype("string")
        self.items_df["item_id"] = self.items_df["item_id"].astype("string")
        if "title" in self.items_df.columns:
            self.items_df["title"] = self.items_df["title"].astype("string")

        self.events_df["watch_seconds"] = pd.to_numeric(
            self.events_df["watch_seconds"], errors="coerce"
        ).fillna(0)
        self.events_df["watch_seconds"] = (
            self.events_df["watch_seconds"].clip(lower=0).round().astype("int64")
        )

        # ---------- Popularity ----------
        pop = (
            self.events_df.groupby("item_id", as_index=False)["watch_seconds"]
            .sum()
            .rename(columns={"watch_seconds": "popularity_watch_seconds"})
        )

        pop = self.items_df[["item_id", "title"]].merge(pop, on="item_id", how="left")
        pop["popularity_watch_seconds"] = (
            pop["popularity_watch_seconds"].fillna(0).astype("int64")
        )
        pop = pop.sort_values(
            ["popularity_watch_seconds", "item_id"], ascending=[False, True]
        ).reset_index(drop=True)

        self._popular_df = pop

        # ---------- User-Item matrix ----------
        ui = (
            self.events_df.groupby(["user_id", "item_id"], as_index=False)["watch_seconds"]
            .sum()
            .rename(columns={"watch_seconds": "watch_seconds_sum"})
        )

        user_item = ui.pivot_table(
            index="user_id",
            columns="item_id",
            values="watch_seconds_sum",
            fill_value=0,
            aggfunc="sum",
        )

        # Ensure all items exist as columns (even if no interactions)
        all_item_ids = self.items_df["item_id"].dropna().astype("string").unique().tolist()
        for iid in all_item_ids:
            if iid not in user_item.columns:
                user_item[iid] = 0

        # Stable order
        user_item = user_item.reindex(sorted(user_item.columns), axis=1)

        self._user_item = user_item
        self._item_ids = user_item.columns.astype("string").tolist()
        self._item_index = {iid: idx for idx, iid in enumerate(self._item_ids)}

        # ---------- Item-item similarity ----------
        # item_user shape: (n_items, n_users)
        item_user = user_item.to_numpy(dtype=np.float32).T
        sim = cosine_similarity(item_user)
        np.fill_diagonal(sim, 0.0)
        self._item_sim = sim

    def recommend_popular(self, k: int, exclude_item_ids: Optional[Iterable[str]] = None) -> list[dict]:
        if self._popular_df is None:
            raise RuntimeError("Model not fitted")

        k = int(k)
        exclude = set(map(str, exclude_item_ids)) if exclude_item_ids else set()

        out: list[dict] = []
        for _, row in self._popular_df.iterrows():
            iid = str(row["item_id"])
            if iid in exclude:
                continue
            out.append(
                {
                    "item_id": iid,
                    "title": str(row.get("title", "")),
                    "score": float(row.get("popularity_watch_seconds", 0)),
                }
            )
            if len(out) >= k:
                break
        return out

    def recommend_for_user(self, user_id: str, k: int) -> RecommendationResult:
        if self._user_item is None or self._item_sim is None or self._item_index is None or self._item_ids is None:
            raise RuntimeError("Model not fitted")

        user_id = str(user_id)
        k = int(k)

        # Cold start: user belum ada di matrix
        if user_id not in self._user_item.index:
            return RecommendationResult(items=self.recommend_popular(k), fallback_used=True)

        user_row = self._user_item.loc[user_id]  # Series indexed by item_id
        if float(user_row.sum()) <= 0:
            return RecommendationResult(items=self.recommend_popular(k), fallback_used=True)

        # Exclude items watched > threshold
        watched_over = set(user_row[user_row > self.watch_exclude_threshold].index.astype("string").tolist())

        # Score: sim @ user_vector
        user_vec = user_row.to_numpy(dtype=np.float32)  # (n_items,)
        scores = self._item_sim @ user_vec              # (n_items,)

        # Apply exclusion
        for iid in watched_over:
            idx = self._item_index.get(iid)
            if idx is not None:
                scores[idx] = -np.inf

        ranked_idx = np.argsort(-scores)
        results: list[dict] = []

        for idx in ranked_idx:
            if len(results) >= k:
                break
            if not np.isfinite(scores[idx]):
                continue
            iid = self._item_ids[idx]
            if iid in watched_over:
                continue
            results.append(
                {
                    "item_id": iid,
                    "title": self._get_title(iid),
                    "score": float(scores[idx]),
                }
            )

        # Top-up dengan popular kalau kurang
        if len(results) < k:
            already = set(r["item_id"] for r in results) | watched_over
            results.extend(self.recommend_popular(k - len(results), exclude_item_ids=already))

        return RecommendationResult(items=results, fallback_used=False)

    def _get_title(self, item_id: str) -> str:
        m = self.items_df["item_id"].astype("string") == str(item_id)
        if not m.any():
            return ""
        return str(self.items_df.loc[m, "title"].iloc[0])

    def _ensure_events_columns(self) -> None:
        required = {"user_id", "item_id", "event_type", "watch_seconds", "timestamp"}
        missing = required - set(self.events_df.columns)
        if missing:
            raise ValueError(f"events_df missing columns: {sorted(missing)}")

    def _ensure_items_columns(self) -> None:
        required = {"item_id", "title"}
        missing = required - set(self.items_df.columns)
        if missing:
            raise ValueError(f"items_df missing columns: {sorted(missing)}")