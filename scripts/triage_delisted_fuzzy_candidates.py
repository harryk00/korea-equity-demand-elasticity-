#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="data/pit_kosdaq/universe/historical_name_recovery/pit_union_all_v4.csv")
    p.add_argument("--candidates", default="data/pit_kosdaq/universe/historical_name_recovery/residual_fuzzy_candidates.csv")
    p.add_argument("--out-dir", default="data/pit_kosdaq/universe/fuzzy_triage_v5")
    p.add_argument("--min-top1", type=float, default=0.96)
    p.add_argument("--min-margin", type=float, default=0.08)
    p.add_argument("--min-length-ratio", type=float, default=0.70)
    return p.parse_args()

def length_ratio(a, b):
    a, b = str(a or ""), str(b or "")
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    u = pd.read_csv(args.universe, dtype={"ticker": str, "corp_code": str})
    u["ticker"] = u["ticker"].astype(str).str.zfill(6)

    c = pd.read_csv(args.candidates, dtype={"ticker": str, "candidate_corp_code": str})
    c["ticker"] = c["ticker"].astype(str).str.zfill(6)
    c["candidate_corp_code"] = c["candidate_corp_code"].astype(str).str.zfill(8)
    c["rank"] = pd.to_numeric(c["rank"], errors="coerce")
    c["similarity"] = pd.to_numeric(c["similarity"], errors="coerce")
    c["length_ratio"] = [length_ratio(a,b) for a,b in zip(c["source_name_key"], c["candidate_name_key"])]

    top1 = c[c["rank"] == 1].copy()
    top2 = c[c["rank"] == 2][["ticker","similarity"]].rename(columns={"similarity":"top2_similarity"})
    top1 = top1.merge(top2, on="ticker", how="left")
    top1["top2_similarity"] = top1["top2_similarity"].fillna(0.0)
    top1["margin"] = top1["similarity"] - top1["top2_similarity"]
    top1["name_containment"] = [
        (str(a) in str(b)) or (str(b) in str(a))
        for a,b in zip(top1["source_name_key"], top1["candidate_name_key"])
    ]

    top1["passes_score"] = (
        (top1["similarity"] >= args.min_top1)
        & (top1["margin"] >= args.min_margin)
        & (top1["length_ratio"] >= args.min_length_ratio)
        & (top1["name_containment"] | (top1["similarity"] >= 0.985))
    )

    counts = top1.loc[top1["passes_score"]].groupby("candidate_corp_code")["ticker"].nunique()
    dup = set(counts[counts > 1].index)
    top1["candidate_unique_across_tickers"] = ~top1["candidate_corp_code"].isin(dup)
    top1["auto_accept"] = top1["passes_score"] & top1["candidate_unique_across_tickers"]

    top1["decision_reason"] = np.select(
        [
            top1["auto_accept"],
            top1["candidate_corp_code"].isin(dup),
            top1["similarity"] < args.min_top1,
            top1["margin"] < args.min_margin,
            top1["length_ratio"] < args.min_length_ratio,
        ],
        [
            "AUTO_ACCEPT_STRICT",
            "REJECT_DUPLICATE_CORP_CODE",
            "REVIEW_LOW_TOP1",
            "REVIEW_SMALL_MARGIN",
            "REVIEW_LENGTH_MISMATCH",
        ],
        default="REVIEW_NAME_STRUCTURE",
    )

    accepted = top1[top1["auto_accept"]].copy()
    mapping = dict(zip(accepted["ticker"], accepted["candidate_corp_code"]))

    patched = u.copy()
    for idx,row in patched.iterrows():
        t = str(row["ticker"]).zfill(6)
        cc = mapping.get(t)
        if not cc:
            continue
        patched.at[idx,"corp_code"] = cc
        patched.at[idx,"corp_code_candidates"] = cc
        patched.at[idx,"corp_code_count"] = 1
        patched.at[idx,"corp_map_status"] = "ok"
        old = str(row.get("corp_map_source") or "").strip()
        srcs = [x for x in [old, "strict_fuzzy_name_v5"] if x]
        patched.at[idx,"corp_map_source"] = "+".join(sorted(set("+".join(srcs).split("+"))))

    patched["eligible_for_pipeline"] = patched["corp_map_status"].eq("ok")
    if "is_spac" in patched.columns:
        patched["eligible_for_pipeline"] &= ~patched["is_spac"].fillna(False)

    residual = set(top1.loc[~top1["auto_accept"],"ticker"])
    manual = c[c["ticker"].isin(residual)].copy()
    manual = manual.merge(top1[["ticker","top2_similarity","margin","decision_reason"]], on="ticker", how="left")

    top1.to_csv(out_dir/"triage_top1.csv", index=False, encoding="utf-8-sig")
    accepted.to_csv(out_dir/"auto_accepted_mappings.csv", index=False, encoding="utf-8-sig")
    manual.sort_values(["ticker","rank"]).to_csv(out_dir/"manual_review_candidates.csv", index=False, encoding="utf-8-sig")
    patched.to_csv(out_dir/"pit_union_all_v5.csv", index=False, encoding="utf-8-sig")
    patched[patched["eligible_for_pipeline"]].to_csv(out_dir/"pit_pipeline_universe_v5.csv", index=False, encoding="utf-8-sig")

    ex = patched[patched["exited_before_end"] == True]
    total = len(ex)
    mapped = int(ex["corp_map_status"].eq("ok").sum())
    missing = int(ex["corp_map_status"].eq("missing").sum())
    coverage = mapped/total if total else float("nan")

    print("\n=== Strict fuzzy triage ===")
    print(f"residual tickers evaluated : {top1['ticker'].nunique()}")
    print(f"strict auto accepted       : {len(accepted)}")
    print(f"still manual review        : {len(residual)}")
    print("\nDecision reasons:")
    print(top1["decision_reason"].value_counts().to_string())

    print("\n=== Delisted mapping coverage after v5 ===")
    print(f"delisted total : {total}")
    print(f"mapped         : {mapped}")
    print(f"missing        : {missing}")
    print(f"coverage       : {coverage:.2%}")
    print("\nDo not run full core if coverage remains <90%.")
    print(f"wrote -> {out_dir}")

if __name__ == "__main__":
    main()
