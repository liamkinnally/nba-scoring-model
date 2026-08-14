
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sqlalchemy import select

from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.data.models import Game, PlayerGameStats
from nba_scoring_model.features.engineering import FeatureEngineer


@dataclass
class TrainingResult:
    target: str
    rows: int
    train_rows: int
    test_rows: int
    cutoff_date: str
    mae: float
    rmse: float
    r2: float
    best_params: Dict[str, object]
    model_path: str


class PlayerStatModel:
    def __init__(
        self,
        target: str,
        db_manager: DatabaseManager,
        feature_engineer: FeatureEngineer,
        model_dir: str | Path,
    ) -> None:
        if target not in {"points", "assists", "rebounds"}:
            raise ValueError(f"Unsupported target: {target}")
        self.target = target
        self.db_manager = db_manager
        self.feature_engineer = feature_engineer
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline: Pipeline | None = None
        self.feature_columns: List[str] = []

    def prepare_training_frame(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        with self.db_manager.get_session() as session:
            rows = list(
                session.execute(
                    select(PlayerGameStats, Game)
                    .join(Game, PlayerGameStats.game_id == Game.game_id)
                    .where(
                        Game.date >= start_date,
                        Game.date <= end_date,
                        PlayerGameStats.minutes_played > 0,
                    )
                    .order_by(Game.date.asc(), Game.game_id.asc(), PlayerGameStats.player_id.asc())
                ).all()
            )

        records = []
        for stat, game in rows:
            features = self.feature_engineer.generate_player_features(
                stat.player_id,
                game.game_id,
                team_id=stat.team_id,
            )
            records.append(
                {
                    "date": game.date,
                    "game_id": game.game_id,
                    "player_id": stat.player_id,
                    self.target: float(getattr(stat, self.target)),
                    **features,
                }
            )

        frame = pd.DataFrame(records)
        if frame.empty:
            raise ValueError("No training rows found for the requested date range")
        return frame.sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)

    @staticmethod
    def chronological_holdout(frame: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dates = np.array(sorted(pd.to_datetime(frame["date"]).dt.normalize().unique()))
        if len(dates) < 5:
            raise ValueError("At least five distinct game dates are required")
        n_test_dates = max(1, int(np.ceil(len(dates) * test_fraction)))
        cutoff = dates[-n_test_dates]
        train = frame[pd.to_datetime(frame["date"]).dt.normalize() < cutoff].copy()
        test = frame[pd.to_datetime(frame["date"]).dt.normalize() >= cutoff].copy()
        if train.empty or test.empty:
            raise ValueError("Chronological split produced an empty partition")
        return train, test

    @staticmethod
    def expanding_date_splits(dates: Sequence[pd.Timestamp], n_splits: int = 4) -> List[Tuple[np.ndarray, np.ndarray]]:
        normalized = pd.Series(pd.to_datetime(dates)).dt.normalize().to_numpy()
        unique_dates = np.array(sorted(pd.unique(normalized)))
        if len(unique_dates) < n_splits + 2:
            n_splits = max(2, len(unique_dates) - 2)
        if n_splits < 2:
            raise ValueError("Not enough distinct dates for time-series cross-validation")

        blocks = np.array_split(unique_dates, n_splits + 1)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for fold in range(1, len(blocks)):
            train_dates = np.concatenate(blocks[:fold])
            val_dates = blocks[fold]
            train_idx = np.flatnonzero(np.isin(normalized, train_dates))
            val_idx = np.flatnonzero(np.isin(normalized, val_dates))
            if len(train_idx) and len(val_idx):
                splits.append((train_idx, val_idx))
        return splits

    def _base_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        )

    def train(self, frame: pd.DataFrame, *, tune: bool = True) -> TrainingResult:
        train_frame, test_frame = self.chronological_holdout(frame)
        excluded = {"date", "game_id", "player_id", self.target}
        self.feature_columns = [c for c in frame.columns if c not in excluded]

        X_train = train_frame[self.feature_columns]
        y_train = train_frame[self.target]
        X_test = test_frame[self.feature_columns]
        y_test = test_frame[self.target]

        pipeline = self._base_pipeline()
        best_params: Dict[str, object] = {}

        if tune:
            cv = self.expanding_date_splits(train_frame["date"], n_splits=3)
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.07],
                "model__max_depth": [2],
                "model__min_samples_leaf": [5],
                "model__subsample": [0.8],
            }
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            self.pipeline = search.best_estimator_
            best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        else:
            self.pipeline = pipeline.fit(X_train, y_train)

        predictions = self.pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        model_path = self.save(
            metadata={
                "target": self.target,
                "feature_columns": self.feature_columns,
                "best_params": best_params,
                "trained_through": str(pd.to_datetime(train_frame["date"]).max()),
            }
        )
        return TrainingResult(
            target=self.target,
            rows=len(frame),
            train_rows=len(train_frame),
            test_rows=len(test_frame),
            cutoff_date=str(pd.to_datetime(test_frame["date"]).min().date()),
            mae=round(mae, 3),
            rmse=round(rmse, 3),
            r2=round(r2, 3),
            best_params=best_params,
            model_path=str(model_path),
        )

    def predict(self, features: Dict[str, float]) -> float:
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained or loaded")
        X = pd.DataFrame([{name: features.get(name, np.nan) for name in self.feature_columns}])
        return float(self.pipeline.predict(X)[0])

    def save(self, metadata: Dict[str, object]) -> Path:
        if self.pipeline is None:
            raise RuntimeError("No model to save")
        path = self.model_dir / f"{self.target}_model.joblib"
        joblib.dump({"pipeline": self.pipeline, "metadata": metadata}, path)
        return path

    def load(self) -> Dict[str, object]:
        path = self.model_dir / f"{self.target}_model.joblib"
        payload = joblib.load(path)
        self.pipeline = payload["pipeline"]
        self.feature_columns = list(payload["metadata"]["feature_columns"])
        return payload["metadata"]
