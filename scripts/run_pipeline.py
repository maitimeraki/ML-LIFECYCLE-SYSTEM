# scripts/initial_training.py
"""
Trains initial champion model at first startup.
Skips if champion already exists.
"""
import sys
from pathlib import Path
from typing import Dict, Any
sys.path.insert(0, "/app")

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import GradientBoostingClassifier


def main():
    from src.registry.model_registry import ModelRegistry, ModelStage
    from src.training.trainer import ModelTrainer, TrainingConfig
    from src.data.processing import ProductionDataProcessor

    MODEL_ID = "customer_churn_model"

    # Skip if champion already exists
    try:
        registry = ModelRegistry(model_id=MODEL_ID)
        champion = registry.get_champion(MODEL_ID)
        if champion:
            print(f"Champion already exists: v{champion.version}. Skipping.")
            return
    except Exception:
        pass

    print("Training initial champion model...")

    # Generate synthetic training data
    rng = np.random.RandomState(42)
    n   = 5000
    df  = pd.DataFrame({
        "feature_1": rng.normal(10, 2, n),
        "feature_2": rng.normal(5, 1.5, n),
        "feature_3": rng.exponential(2, n),
        "feature_4": rng.choice(["A", "B", "C", "D"], n),
        "feature_5": rng.uniform(0, 100, n),
    })
    logits       = 0.3 * df["feature_1"] - 0.5 * df["feature_2"]
    df["target"] = (logits > logits.median()).astype(int)

    # Process data
    processor = ProductionDataProcessor(
        model_id=MODEL_ID,
        pipeline_run_id="initial",
        target_column="target",
    )
    result = processor.fit_transform(
        reference_df=df,
        production_df=df,
        save_path=f"artifacts/processors/{MODEL_ID}/processor.joblib",
    )

    # Save reference data
    Path(f"artifacts/reference/{MODEL_ID}").mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        f"artifacts/reference/{MODEL_ID}/reference_raw.parquet",
        index=False,
    )

    # Train model
    feature_cols = [
        c for c in result.production_processed.columns
        if c != "target"
    ]
    config = TrainingConfig(
        model_type="gradient_boosting",
        hyperparameters={
            "n_estimators": 100,
            "max_depth":    5,
            "learning_rate": 0.1,
            "random_state":  42,
        },
        feature_columns=feature_cols,
        target_column="target",
        cv_folds=3,
        cv_scoring="f1_weighted",
    )
    def model_factory(hp: Dict[str, Any])->Any:
            return GradientBoostingClassifier(**hp)

    trainer = ModelTrainer(
        model_factory=model_factory,
        model_id=MODEL_ID,
        pipeline_run_id="initial",
    )

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    training_result = trainer.train(
        df=result.production_processed,
        config=config,
        model_id=MODEL_ID,
        model_version=version,
    )

    # Register and promote to champion
    registry = ModelRegistry(model_id=MODEL_ID)
    registry.register(
        model_id=MODEL_ID,
        version=version,
        mlflow_run_id=training_result.mlflow_run_id,
        metrics=training_result.metrics,
        hyperparameters=config.hyperparameters,
        data_hash=training_result.data_hash,
        artifact_path=training_result.model_artifact_path,
        stage=ModelStage.STAGING,
        tags={"initial_training": "true"},
    )
    registry.promote_to_champion(
        model_id=MODEL_ID,
        version=version,
    )

    print(f"Champion registered: v{version}")
    print(f"Metrics: {training_result.metrics}")


if __name__ == "__main__":
    main()