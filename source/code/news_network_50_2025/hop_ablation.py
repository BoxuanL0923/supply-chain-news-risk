"""Cumulative 1/2/3-hop scopes and matched, exploratory risk regressions.

The same policy, controls, cases, and forecast target must be used at every
radius. These are predictive contrasts, not causal shock-transmission effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from news_exposure_2025.metrics import regression

RADII = (1, 2, 3)
TERMS = ("H1", "H2_minus_H1", "H3_minus_H2")


def matched_scores(cases: list[dict], controls: pd.DataFrame, labels: pd.DataFrame,
                   records: dict[int, list[dict]], own_records: list[dict] | None = None) -> pd.DataFrame:
    """Join all radii on exactly the same case IDs; never silently drop rows."""
    expected = [case["case_id"] for case in cases]
    if len(expected) != len(set(expected)) or set(records) != set(RADII):
        raise ValueError("Invalid case/radius set")
    for table in (controls, labels):
        if ("case_id" not in table or len(table) != len(expected)
                or set(table.case_id) != set(expected) or table.case_id.duplicated().any()):
            raise ValueError("Incomplete controls or labels")
    frame = controls.set_index("case_id").loc[expected].join(labels.set_index("case_id")[["y"]])
    for radius in RADII:
        rows = records[radius]
        ids = [row["case_id"] for row in rows]
        if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError("Incomplete or duplicated radius score")
        frame[f"H{radius}"] = pd.Series({row["case_id"]: row["score"] for row in rows}).loc[expected]
    if own_records is not None:
        ids = [row["case_id"] for row in own_records]
        if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError("Incomplete or duplicated OWN score")
        frame["OWN"] = pd.Series({row["case_id"]: row["score"] for row in own_records}).loc[expected]
    numeric = ["y", "H1", "H2", "H3"] + (["OWN"] if own_records is not None else [])
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite radius score or outcome")
    return frame.reset_index()


def scope_network_pool(pool: dict, documents: dict, radius: int, *,
                       require_radius_three: bool = False):
    """Return a new typed-path capability without deeper or co-mention leads."""
    if type(radius) is not int or radius not in RADII:
        raise ValueError("Radius must be 1, 2, or 3")
    relations = pool.get("relations")
    if not isinstance(relations, list) or not isinstance(pool.get("network_ids"), list):
        raise ValueError("Malformed network pool")
    typed = []
    for path in relations:
        if not isinstance(path, dict):
            raise ValueError("Malformed path")
        if path.get("type") == "comention":
            continue
        hops = path.get("hops")
        if type(hops) is not int or hops not in RADII or not isinstance(path.get("permnos"), list):
            raise ValueError("Malformed typed path")
        if hops <= radius:
            typed.append(path)
    if require_radius_three and not any(p["hops"] == 3 for p in relations):
        raise ValueError("Three-hop candidate pool required")
    reachable = {p for path in typed for p in path["permnos"]}
    ids = []
    for sid in pool["network_ids"]:
        doc = documents.get(sid)
        if not isinstance(doc, dict) or not isinstance(doc.get("ids"), list):
            raise ValueError("Missing scoped document")
        if set(doc["ids"]) & reachable:
            ids.append(sid)
    # The original 31-day count was computed over the full three-hop pool.
    # Do not leak deeper-hop coverage into a narrower prompt.
    coverage = {k: v for k, v in pool.get("coverage", {}).items()
                if k not in {"network_31d_count", "unmapped_paths"}}
    coverage = {**coverage, "max_hops": radius,
                "typed_relations": len(typed), "network_candidates": len(ids),
                "unmapped_paths": sum(not path["permnos"] for path in typed),
                "comention_excluded": True}
    return ids, typed, coverage


def exact_hop_counts(items: list[dict], relations: list[dict], *, document_mode=False) -> dict[int, int]:
    """Count sources/events by shortest typed path, excluding co-mentions."""
    counts = {h: 0 for h in RADII}
    for item in items:
        paths = ([p for p in relations if set(p["permnos"]) & set(item["ids"])]
                 if document_mode else item.get("verified_paths", []))
        eligible = [p["hops"] for p in paths if p.get("type") != "comention" and p.get("hops") in RADII]
        if eligible:
            counts[min(eligible)] += 1
    return counts


def regression_report(frame: pd.DataFrame, *, controls: tuple[str, ...],
                      train: pd.DataFrame | None = None) -> dict:
    """Nested OLS and optional train-fitted test readouts on complete rows.

    H1 is the radius-1 score. Incremental columns are changes in that score
    when the radius expands; a nested coefficient is predictive, not causal.
    """
    required = ["y", *controls, "H1", "H2", "H3"]
    for sample in (frame, train) if train is not None else (frame,):
        if sample is None or any(column not in sample for column in required):
            raise ValueError("Missing regression column")
        if not np.isfinite(sample[required].to_numpy(float)).all():
            raise ValueError("Incomplete/nonfinite matched rows")
    def design(sample, radius):
        base = np.c_[np.ones(len(sample)), sample[list(controls)].to_numpy(float)]
        columns = [sample.H1.to_numpy(float),
                   (sample.H2 - sample.H1).to_numpy(float),
                   (sample.H3 - sample.H2).to_numpy(float)]
        return np.c_[base, *columns[:radius]] if radius else base

    base = regression(frame.y, design(frame, 0))
    nested = {}
    for radius in RADII:
        fitted = regression(frame.y, design(frame, radius))
        previous = base if radius == 1 else nested[f"through_h{radius-1}"]
        item = {"r2": fitted["r2"], "rank": fitted["rank"],
                "increment_pp": 100 * (fitted["r2"] - previous["r2"]),
                "rank_increment": fitted["rank"] - previous["rank"]}
        item["coefficient"] = fitted["coefficient"][-1] if item["rank_increment"] else None
        if train is not None:
            trained = regression(train.y, design(train, radius))
            trained_base = regression(train.y, design(train, 0))
            trained_previous = trained_base if radius == 1 else regression(train.y, design(train, radius - 1))
            y = frame.y.to_numpy(float)
            e = y - design(frame, radius) @ np.asarray(trained["coefficient"])
            e0 = y - design(frame, 0) @ np.asarray(trained_base["coefficient"])
            item.update(frozen_mse=float(np.mean(e**2)),
                        frozen_r2_vs_base=float(1 - np.sum(e**2)/np.sum(e0**2)),
                        training_rank_increment=trained["rank"] - trained_previous["rank"])
        nested[f"through_h{radius}"] = item
    return {"n": len(frame), "terms": list(TERMS),
            "base_r2": base["r2"], "nested": nested,
            "interpretation": "H2-H1 and H3-H2 are incremental forecast-score changes on identical cases; coefficients and R2 differences are predictive, not causal."}


def execute(out):
    """Run fixed-policy radius arms before opening test labels; write matched outputs.

    This deliberately uses the existing provider and source-review gates. It
    requires a separately prepared three-hop directory and does not reuse the
    historical two-hop experiment as if it contained third-hop news.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from filelock import FileLock
    from news_agent_batched_runtime_2025 import digest, read, write
    from news_agent_r2_2025.config import CLAUDE
    from .config import CONTROLS, SEED, WORKERS
    from .metrics import anchor
    from .run import preflight, resource_limits
    from .runtime import NativeClient
    from .workflow import Runner, pool_file

    if read(out / "data_manifest.json").get("max_hops") != 3:
        raise ValueError("Prepare a separate --max-hops 3 output directory first")
    result_dir = out / "hop_ablation"
    result_dir.mkdir(exist_ok=True)
    with FileLock(str(result_dir / ".run.lock"), timeout=0):
        if (result_dir / "summary.json").exists():
            raise ValueError("Hop ablation already complete; do not overwrite reported results")
        cases, manifest = preflight(out)
        review = read(out / "toolkit_review.json")
        if not review.get("approved") or review.get("manifest") != digest(manifest):
            raise ValueError("Reviewed native graph-tool capability required")
        has_third = any(any(p.get("hops") == 3 for p in
                            pool_file(str(out), case["date"])["pools"][case["case_id"]]["relations"])
                        for case in cases)
        if not has_third:
            raise ValueError("No third-hop graph paths in prepared pool")
        protocol = {"manifest": digest(manifest), "policy_sha256": digest(SEED),
                    "radii": list(RADII), "arms": ["OWN", "H1", "H2", "H3"],
                    "score_contract": "identical fixed seed, cases, controls and model; typed paths only",
                    "regression_terms": list(TERMS), "test_labels_after_prediction_freeze": True}
        protocol_path = result_dir / "PROTOCOL.json"
        if protocol_path.exists() and read(protocol_path) != protocol:
            raise ValueError("Hop ablation protocol changed during resume")
        write(protocol_path, protocol)
        anchors = anchor(out, cases)
        client = NativeClient(out, resource_limits(out), CLAUDE)
        client.semaphore = threading.BoundedSemaphore(WORKERS)
        client.budget.start()
        runners = {radius: Runner(out, client, cases, anchors, digest(manifest),
                                  digest(manifest), max_hops=radius) for radius in RADII}
        own_runner = runners[1]
        selected = [case for case in cases if case["split"] in ("train", "test")]

        def panel(runner, batch, network):
            with ThreadPoolExecutor(max_workers=WORKERS) as workers:
                return list(workers.map(lambda case: runner.score(case, SEED, network), batch))

        own = panel(own_runner, selected, False)
        records = {radius: panel(runners[radius], selected, True) for radius in RADII}
        # Freeze *all* test predictions and the fixed policy before reading any
        # test outcomes. The output is useful for later independent analyses.
        write(result_dir / "prediction_keys.json", {
            "policy": "fixed_seed", "manifest": digest(manifest),
            "case_ids": [case["case_id"] for case in selected],
            "OWN": [row["key"] for row in own],
            **{f"H{radius}": [row["key"] for row in records[radius]] for radius in RADII}})

        by_radius = {r: {row["case_id"]: row for row in records[r]} for r in RADII}
        scope_audit = {}
        for case in selected:
            case_id = case["case_id"]
            day = pool_file(str(out), case["date"])
            pool = day["pools"][case_id]
            audit = {f"candidate_exact_h{h}": count for h, count in
                     exact_hop_counts([day["documents"][sid] for sid in pool["network_ids"]],
                                      pool["relations"], document_mode=True).items()}
            for radius in RADII:
                key = by_radius[radius][case_id]["network_evidence_key"]
                evidence = read(out / "evidence" / (key + ".json"))
                counts = exact_hop_counts(evidence["events"], [])
                audit.update({f"H{radius}_accepted_exact_h{h}": count for h, count in counts.items()})
            scope_audit[case_id] = audit
        write(result_dir / "scope_audit.json", scope_audit)

        controls = pd.read_parquet(out / "controls.parquet")
        frames = {}
        for split in ("train", "test"):
            split_cases = [case for case in selected if case["split"] == split]
            ids = {case["case_id"] for case in split_cases}
            labels = pd.read_parquet(out / f"labels_{split}.parquet")[["case_id", "y"]]
            frames[split] = matched_scores(split_cases, controls[controls.case_id.isin(ids)],
                labels, {r: [row for row in records[r] if row["case_id"] in ids] for r in RADII},
                [row for row in own if row["case_id"] in ids])
            frames[split]["anchor"] = frames[split].case_id.map(anchors)
            for column in next(iter(scope_audit.values())):
                frames[split][column] = frames[split].case_id.map(
                    {case_id: audit[column] for case_id, audit in scope_audit.items()})
        train, test = frames["train"], frames["test"]
        control_columns = (*CONTROLS, "OWN")
        report = {"protocol": "fixed policy; cumulative typed paths; same cases, controls, target and provider model",
                  "train": regression_report(train, controls=control_columns),
                  "test": regression_report(test, controls=control_columns, train=train),
                  "raw_residual_mse": {name: float(np.mean((test.y-test.anchor-test[name])**2))
                                       for name in ("OWN", "H1", "H2", "H3")},
                  "test_source_coverage": {column: int(test[column].sum()) for column in
                      ("candidate_exact_h1", "candidate_exact_h2", "candidate_exact_h3",
                       "H1_accepted_exact_h1", "H2_accepted_exact_h2", "H3_accepted_exact_h3")},
                  "limitations": "Path-only evidence is not product continuity; three-hop forecasting is conditional on the prepared pool and reviewed retrieval, with exploratory inference."}
        test.to_csv(result_dir / "matched_test.csv", index=False)
        write(result_dir / "summary.json", report)
        lines = ["# Supply-chain hop ablation", "",
                 "Fixed seed policy, identical firm-periods and controls, cumulative typed paths. Co-mentions are excluded.",
                 "H1 uses one hop; H2 uses up to two; H3 uses up to three. H2-H1 and H3-H2 are forecast-score changes.", "",
                 "| Radius | Raw residual MSE | Pooled incremental R2 (pp) | Frozen R2 vs OWN + controls |",
                 "|---|---:|---:|---:|"]
        for radius in RADII:
            item = report["test"]["nested"][f"through_h{radius}"]
            lines.append(f"| H{radius} | {report['raw_residual_mse'][f'H{radius}']:.5f} | "
                         f"{item['increment_pp']:.3f} | {item['frozen_r2_vs_base']:.4f} |")
        lines += ["", "These nested regression increments are predictive associations, not causal transmission.",
                  "Inspect summary.json test_source_coverage to see whether each radius actually accepted news at its newly allowed hop.",
                  "A zero rank increment means the corresponding hop contrast is unidentified in this sample.",
                  "Test labels were opened only after all fixed-policy prediction keys were written."]
        (result_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from .config import OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    execute(args.out)
