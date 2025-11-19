from pathlib import Path
from datetime import timedelta

import pandas as pd

# Reuse the AI commentary function you already built
from generate_ai_commentary import generate_commentary

# Figure out project root (same trick as before)
project_root = Path(__file__).resolve().parent.parent

# 1) Load the full sector-level sample data
data_path = project_root / "data" / "sector_attribution_sample.csv"
print("Loading sector data from:", data_path)

df = pd.read_csv(data_path, parse_dates=["date"])

# 2) Define the reporting period: last 30 days in the dataset
end_date = df["date"].max()
start_date = end_date - pd.Timedelta(days=30)

period_df = df[df["date"].between(start_date, end_date)].copy()

if period_df.empty:
    raise RuntimeError("No data found in the last 30 days of the sample dataset.")

# 3) Build sector-level summary for that period
sector_summary = (
    period_df.groupby("sector")[["allocation_effect", "selection_effect", "interaction_effect", "excess_return"]]
    .sum()
    .sort_values("excess_return", ascending=False)
)

print("\nSector summary for period:")
print(sector_summary)

# 4) Identify top contributors and detractors
top_positive = sector_summary["excess_return"].nlargest(3)
top_negative = sector_summary["excess_return"].nsmallest(3)

def format_top(series):
    lines = []
    for sector, value in series.items():
        lines.append(f"- {sector}: {value:.4f}")
    return "\n".join(lines)

top_pos_text = format_top(top_positive)
top_neg_text = format_top(top_negative)

# 5) Get AI commentary using your existing function
ai_commentary = generate_commentary(sector_summary)

# 6) Build a simple monthly-style report (Markdown)
report_text = f"""# Monthly Performance Attribution Summary

**Period:** {start_date.date()} to {end_date.date()}

## Overview

This report summarizes sector-level performance attribution for the most recent 30-day period in the dataset. Excess return is measured as portfolio return minus benchmark return, decomposed into allocation, selection, and interaction effects.

## Top Contributing Sectors (by Excess Return)

{top_pos_text or "No positive contributors in this period."}

## Largest Detracting Sectors (by Excess Return)

{top_neg_text or "No negative detractors in this period."}

## AI-Generated Attribution Commentary

{ai_commentary}
"""

# 7) Save report to outputs/
outputs_dir = project_root / "outputs"
outputs_dir.mkdir(exist_ok=True)

report_path = outputs_dir / "monthly_report.md"
with open(report_path, "w") as f:
    f.write(report_text)

print(f"\nMonthly report saved to: {report_path}")