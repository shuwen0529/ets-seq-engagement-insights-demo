from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def _pct(x: float) -> str:
    return f"{x*100:.1f}%"

def summarize_fatigue(fatigue_df: pd.DataFrame) -> str:
    df = fatigue_df.copy().sort_values("events_last_window")
    peak_row = df.loc[df["actual_rate"].idxmax()]
    peak_k = int(peak_row["events_last_window"])
    peak_rate = float(peak_row["actual_rate"])

    three = df[df["events_last_window"] == 3]
    five_plus = df[df["events_last_window"] >= 5]

    if not three.empty and not five_plus.empty:
        r3 = float(three["actual_rate"].iloc[0])
        r5p = float(five_plus["actual_rate"].mean())
        delta_pp = (r5p - r3) * 100
        return (
            f"Observed **engagement fatigue**: users with **≥5 events in the last 14 days** "
            f"had an outcome rate of ~{_pct(r5p)} vs ~{_pct(r3)} for **3 events** "
            f"({delta_pp:+.1f} pp)."
        )

    last_row = df.iloc[-1]
    last_k = int(last_row["events_last_window"])
    last_rate = float(last_row["actual_rate"])
    delta_pp = (last_rate - peak_rate) * 100
    return (
        f"Observed **fatigue/saturation pattern**: outcome rate peaks around **{peak_k} events** "
        f"(~{_pct(peak_rate)}) and shifts to ~{_pct(last_rate)} at **{last_k} events** "
        f"({delta_pp:+.1f} pp)."
    )

def summarize_gap_sensitivity(sweep_df: pd.DataFrame) -> str:
    df = sweep_df.copy().sort_values("gap_days_set_at_last_step")
    g_min = int(df["gap_days_set_at_last_step"].min())
    g_max = int(df["gap_days_set_at_last_step"].max())
    p_min = float(df.loc[df["gap_days_set_at_last_step"] == g_min, "avg_pred_prob"].iloc[0])
    p_max = float(df.loc[df["gap_days_set_at_last_step"] == g_max, "avg_pred_prob"].iloc[0])
    delta_pp = (p_max - p_min) * 100
    return (
        f"**Timing sensitivity**: when the most recent gap increases from **{g_min}d** to **{g_max}d**, "
        f"average predicted probability shifts from ~{_pct(p_min)} to ~{_pct(p_max)} "
        f"({delta_pp:+.1f} pp)."
    )

def summarize_counterfactual(insights_json: dict) -> str:
    cf = insights_json.get("counterfactual_add_recent_event", {})
    base = float(cf.get("base_avg_pred", float("nan")))
    after = float(cf.get("after_add_recent_content_review_gap2d", float("nan")))
    delta = float(cf.get("delta", float("nan")))
    if base != base or after != after or delta != delta:
        return "Counterfactual insight fields missing; re-run `scripts/04_extract_insights.py`."
    return (
        f"**Counterfactual test**: adding a recent **content_review** event (gap=2d) changes "
        f"average predicted probability from ~{_pct(base)} to ~{_pct(after)} "
        f"({delta*100:+.1f} pp)."
    )

def main():
    repo_root = Path(__file__).resolve().parents[1]
    tables_dir = repo_root / "outputs" / "tables"
    metrics_dir = repo_root / "outputs" / "metrics"

    fatigue_df = pd.read_csv(tables_dir / "fatigue_cohort_14d.csv")
    sweep_df = pd.read_csv(tables_dir / "time_gap_sensitivity.csv")
    with open(metrics_dir / "insights_summary.json", "r", encoding="utf-8") as f:
        insights_json = json.load(f)

    bullets = [
        "### Slide 4 — Behavioral Insights (Evidence-Based)",
        "",
        "**How insights were derived:**",
        "- Cohort analytics on event logs (observed outcome rate patterns)",
        "- Counterfactual sensitivity tests on the trained sequence model (what-if deltas)",
        "",
        "**Key insights:**",
        f"- {summarize_fatigue(fatigue_df)}",
        f"- {summarize_gap_sensitivity(sweep_df)}",
        f"- {summarize_counterfactual(insights_json)}",
        "",
        "**Operational implications (ETS framing):**",
        "- Calibrate cadence to avoid saturation and improve completion / re-engagement likelihood.",
        "- Use timing-aware nudges: intervene within ‘momentum windows’ and avoid long inactivity gaps.",
        "- Segment-aware policies: fatigue and timing sensitivity can differ by user group.",
        "",
        "_Figures to paste into Slide 4:_",
        "- `outputs/figures/fatigue_cohort.png`",
        "- `outputs/figures/time_gap_sensitivity.png`",
    ]

    (tables_dir / "slide4_summary.md").write_text("\n".join(bullets) + "\n", encoding="utf-8")
    (tables_dir / "slide4_summary.txt").write_text("\n".join([b.replace("**", "") for b in bullets]) + "\n", encoding="utf-8")

    print("Wrote slide-ready summaries:")
    print("-", tables_dir / "slide4_summary.md")
    print("-", tables_dir / "slide4_summary.txt")

if __name__ == "__main__":
    main()
