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

        item_cols = ["item_id", "title"]
        for c in ["content_type", "genre"]:
            if c in self.items_df.columns:
                item_cols.append(c)
        pop = self.items_df[item_cols].merge(pop, on="item_id", how="left")
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
        # Tambahkan kolom yang hilang sekaligus (lebih cepat, tidak fragmented)
        user_item = user_item.reindex(columns=all_item_ids, fill_value=0.0)

        # Stable order
        user_item = user_item.reindex(sorted(user_item.columns), axis=1)

        self._user_item = user_item
        self._item_ids = user_item.columns.astype("string").tolist()
        self._item_index = {iid: idx for idx, iid in enumerate(self._item_ids)}

        # ---------- Item-item similarity ----------
        # item_user shape: (n_items, n_users)
        item_user = user_item.to_numpy(dtype=np.float32).T
        n_items, n_users = item_user.shape
        if n_items == 0:
            sim = np.zeros((0, 0), dtype=np.float32)
        elif n_users == 0:
            # No users / no interactions -> similarity matrix all zeros
            sim = np.zeros((n_items, n_items), dtype=np.float32)
        else:
            sim = cosine_similarity(item_user)

        if sim.size:
            np.fill_diagonal(sim, 0.0)
        self._item_sim = sim

    def recommend_popular(
        self,
        k: int,
        exclude_item_ids: Optional[Iterable[str]] = None,
        content_type: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> list[dict]:
        if self._popular_df is None:
            raise RuntimeError("Model not fitted")

        k = int(k)
        exclude = set(map(str, exclude_item_ids)) if exclude_item_ids else set()
        df = self._popular_df
        if content_type and "content_type" in df.columns:
            ct = str(content_type).strip().lower()
            df = df[df["content_type"].astype("string").str.lower() == ct]
        if genre and "genre" in df.columns:
            g = str(genre).strip().lower()
            df = df[df["genre"].astype("string").str.lower() == g]

        out: list[dict] = []
        for _, row in df.iterrows():
            iid = str(row["item_id"])
            if iid in exclude:
                continue
            out.append(
                {
                    "item_id": iid,
                    "title": str(row.get("title", "")),
                    "score": float(row.get("popularity_watch_seconds", 0)),
                    "reason": "popular",
                }
            )
            if len(out) >= k:
                break
        return out

    def recommend_for_user(
        self,
        user_id: str,
        k: int,
        content_type: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> RecommendationResult:
        if self._user_item is None or self._item_sim is None or self._item_index is None or self._item_ids is None:
            raise RuntimeError("Model not fitted")

        user_id = str(user_id)
        k = int(k)

        def _tag_popular_reason(items: list[dict]) -> list[dict]:
            # sudah ada reason dari recommend_popular, tapi ini biar aman
            return [{**d, "reason": d.get("reason", "popular")} for d in items]

        allowed_idx: Optional[set[int]] = None
        if content_type or genre:
            m = pd.Series(True, index=self.items_df.index)
            if content_type and "content_type" in self.items_df.columns:
                ct = str(content_type).strip().lower()
                m &= self.items_df["content_type"].astype("string").str.lower() == ct
            if genre and "genre" in self.items_df.columns:
                g = str(genre).strip().lower()
                m &= self.items_df["genre"].astype("string").str.lower() == g
            allowed_ids = self.items_df.loc[m, "item_id"].astype("string").tolist()
            allowed_idx = {self._item_index[iid] for iid in allowed_ids if iid in self._item_index}

        # Cold start: user belum ada di matrix
        if user_id not in self._user_item.index:
            return RecommendationResult(
                items=_tag_popular_reason(self.recommend_popular(k, content_type=content_type, genre=genre)),
                fallback_used=True,
            )

        user_row = self._user_item.loc[user_id]  # Series indexed by item_id
        if float(user_row.sum()) <= 0:
            return RecommendationResult(
                items=_tag_popular_reason(self.recommend_popular(k, content_type=content_type, genre=genre)),
                fallback_used=True,
            )

        # Exclude items watched > threshold
        watched_over = set(user_row[user_row > self.watch_exclude_threshold].index.astype("string").tolist())

        # Score: sim @ user_vector
        user_vec = user_row.to_numpy(dtype=np.float32)  # (n_items,)
        scores = self._item_sim @ user_vec              # (n_items,)

        watched_idx = np.where(user_vec > 0)[0]

        def _reason_for_idx(item_idx: int) -> str | None:
            # Cari watched item yang paling mirip sebagai "alasan"
            if watched_idx.size == 0:
                return None
            sims = self._item_sim[item_idx, watched_idx]
            if sims.size == 0:
                return None
            best_pos = int(np.argmax(sims))
            best_sim = float(sims[best_pos])
            if best_sim <= 0:
                return None
            seed_idx = int(watched_idx[best_pos])
            seed_item_id = self._item_ids[seed_idx]
            seed_title = self._get_title(seed_item_id)
            if not seed_title:
                return None
            return f"similar to item you watched: {seed_title}"

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
            if allowed_idx is not None and int(idx) not in allowed_idx:
                continue
            iid = self._item_ids[idx]
            if iid in watched_over:
                continue
            results.append(
                {
                    "item_id": iid,
                    "title": self._get_title(iid),
                    "score": float(scores[idx]),
                    "reason": _reason_for_idx(int(idx)),
                }
            )

        # Top-up dengan popular kalau kurang
        if len(results) < k:
            already = set(r["item_id"] for r in results) | watched_over
            topup = self.recommend_popular(
                k - len(results),
                exclude_item_ids=already,
                content_type=content_type,
                genre=genre,
            )
            results.extend(_tag_popular_reason(topup))

        return RecommendationResult(items=results, fallback_used=False)

    def _get_title(self, item_id: str) -> str:
        m = self.items_df["item_id"].astype("string") == str(item_id)
        if not m.any():
            return ""
        return str(self.items_df.loc[m, "title"].iloc[0])

    def get_user_history(self, user_id: str, k: int = 20) -> list[dict]:
        """
        Return top-k watch history for a user (most recent first),
        including item_id, title, watch_seconds, timestamp.
        If user not found / no history -> empty list.
        """
        user_id = str(user_id)
        k = int(k)
        if k <= 0:
            return []

        if "user_id" not in self.events_df.columns:
            return []

        df = self.events_df[self.events_df["user_id"].astype("string") == user_id].copy()
        if df.empty:
            return []

        # Pastikan timestamp bisa disort (kalau sudah datetime dari DataLoader, ini aman)
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            except Exception:
                pass

        # Ambil event paling recent per item_id, dan total watch_seconds per item_id
        agg = df.groupby("item_id", as_index=False).agg(
            watch_seconds=("watch_seconds", "sum"),
            last_timestamp=("timestamp", "max"),
        )
        agg = agg.sort_values(["last_timestamp", "watch_seconds"], ascending=[False, False]).head(k)

        out: list[dict] = []
        for _, row in agg.iterrows():
            iid = str(row["item_id"])
            ts = row.get("last_timestamp")
            out.append(
                {
                    "item_id": iid,
                    "title": self._get_title(iid),
                    "watch_seconds": int(row.get("watch_seconds", 0)),
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") and pd.notna(ts) else None,
                }
            )
        return out

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