#!/usr/bin/env python3
"""Train tree-based MS2Forge rerankers with a shared evaluation protocol.

Backends:
  lightgbm: LambdaMART / LGBMRanker
  xgboost: XGBRanker if xgboost is installed
  catboost: CatBoostRanker if catboost is installed
  sklearn: pointwise HistGradientBoosting baseline
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from rerank import training_features as fut
from rerank import training_common as tc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-table", required=True)
    p.add_argument("--valid-table", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--backend", choices=["lightgbm", "xgboost", "catboost", "sklearn"], default="lightgbm")
    p.add_argument("--features", default=None, help="Comma-separated feature columns. Default: archived 241-feature EvidenceRank contract.")
    p.add_argument("--feature-preset", choices=sorted(fut.FEATURE_PRESETS), default=None)
    p.add_argument("--include-group-relative", action="store_true")
    p.add_argument("--group-order", choices=["sort", "check", "none"], default="sort")
    p.add_argument("--truth-column", default="candidate_is_true")
    p.add_argument("--positive-groups-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-train-groups", type=int, default=None)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--colsample", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--xgb-objective", choices=["rank:ndcg", "rank:pairwise"], default="rank:ndcg")
    p.add_argument("--cat-loss", choices=["YetiRank", "PairLogit"], default="YetiRank")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--topk", default="1,5,10")
    recipe = tc.recipe("evidencerank")
    p.set_defaults(**recipe["parameters"], features=",".join(recipe["features"]))
    tc.add_data_arguments(p)
    return p.parse_args()


def train_lightgbm(args: argparse.Namespace, train_rows: list[dict[str, Any]], features: list[str]):
    import lightgbm as lgb

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth if args.max_depth > 0 else -1,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        reg_lambda=args.l2,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    model.fit(
        fut.matrix(train_rows, features),
        fut.labels(train_rows, args.truth_column),
        group=fut.group_sizes(train_rows),
    )
    return model


def train_xgboost(args: argparse.Namespace, train_rows: list[dict[str, Any]], features: list[str]):
    try:
        import xgboost as xgb
    except Exception as exc:
        raise SystemExit(
            "xgboost is not installed in this environment. Install it or choose --backend lightgbm/sklearn."
        ) from exc

    model = xgb.XGBRanker(
        objective=args.xgb_objective,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        reg_lambda=args.l2,
        random_state=args.seed,
        tree_method="hist",
        n_jobs=args.n_jobs,
        eval_metric="ndcg@10",
    )
    model.fit(
        fut.matrix(train_rows, features),
        fut.labels(train_rows, args.truth_column),
        group=fut.group_sizes(train_rows),
        verbose=False,
    )
    return model


def train_catboost(args: argparse.Namespace, train_rows: list[dict[str, Any]], features: list[str]):
    try:
        from catboost import CatBoostRanker, Pool
    except Exception as exc:
        raise SystemExit(
            "catboost is not installed in this environment. Install it or choose --backend lightgbm/sklearn."
        ) from exc

    pool = Pool(
        data=fut.matrix(train_rows, features),
        label=fut.labels(train_rows, args.truth_column),
        group_id=fut.group_ids(train_rows),
        feature_names=features,
    )
    model = CatBoostRanker(
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.max_depth,
        l2_leaf_reg=args.l2 if args.l2 > 0 else 3.0,
        loss_function=args.cat_loss,
        random_seed=args.seed,
        thread_count=args.n_jobs,
        verbose=100,
    )
    model.fit(pool)
    return model


def train_sklearn(args: argparse.Namespace, train_rows: list[dict[str, Any]], features: list[str]):
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = fut.labels(train_rows, args.truth_column)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    pos_weight = min(neg / pos, 100.0)
    sample_weight = [pos_weight if label > 0 else 1.0 for label in y]
    model = HistGradientBoostingClassifier(
        max_iter=args.n_estimators,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.num_leaves,
        l2_regularization=args.l2,
        random_state=args.seed,
    )
    model.fit(fut.matrix(train_rows, features), y, sample_weight=sample_weight)
    return model


def predict_scores(model: Any, rows: list[dict[str, Any]], features: list[str]) -> list[float]:
    x = fut.matrix(rows, features)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return [float(v[1]) for v in proba]
    return [float(v) for v in model.predict(x)]


def feature_importance(model: Any, features: list[str]) -> list[tuple[str, float]]:
    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        values = model.get_feature_importance()
    if values is None:
        return []
    return sorted(zip(features, [float(v) for v in values]), key=lambda item: item[1], reverse=True)


def main() -> None:
    args = parse_args()
    tc.validate_training_inputs(args)
    fut.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topk = [int(x) for x in args.topk.split(",") if x.strip()]

    train_rows_all = fut.prepare_grouped_rows(fut.load_rows(args.train_table), args.group_order)
    valid_rows = fut.prepare_grouped_rows(fut.load_rows(args.valid_table), args.group_order)
    train_rows = fut.filter_groups(
        train_rows_all,
        positive_only=args.positive_groups_only,
        max_groups=args.max_train_groups,
        truth_column=args.truth_column,
    )
    if args.feature_preset:
        raise ValueError("Use --features for an explicit feature list; the default is the archived 241-feature contract")
    features = fut.select_features(train_rows, args.features, args.feature_preset, args.include_group_relative)
    tc.validate_features(train_rows, features)
    tc.validate_features(valid_rows, features)
    print(
        f"[tree] backend={args.backend} train_rows={len(train_rows)} "
        f"valid_rows={len(valid_rows)} features={len(features)}"
    )

    trainers = {
        "lightgbm": train_lightgbm,
        "xgboost": train_xgboost,
        "catboost": train_catboost,
        "sklearn": train_sklearn,
    }
    model = trainers[args.backend](args, train_rows, features)

    valid_scores = predict_scores(model, valid_rows, features)
    fut.write_scores(out_dir / "valid_scores.tsv", valid_rows, valid_scores)
    valid_metrics = fut.evaluate_from_scores(valid_rows, valid_scores, f"{args.backend}_valid", topk)

    metadata = {
        "script": Path(__file__).name,
        "backend": args.backend,
        "args": vars(args),
        "train_table": args.train_table,
        "valid_table": args.valid_table,
        "n_train_rows_all": len(train_rows_all),
        "n_train_rows_used": len(train_rows),
        "n_valid_rows": len(valid_rows),
        "features": features,
        "valid_metrics": valid_metrics,
        "feature_importance": feature_importance(model, features),
        "selection_policy": "Select models on validation data only; evaluate frozen models separately.",
    }
    fut.dump_json(out_dir / "metrics.json", metadata)
    with (out_dir / "evidencerank_full_model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "features": features, "metadata": metadata}, handle)
    print(json.dumps({"valid_metrics": valid_metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
