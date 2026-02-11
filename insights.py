"""import pandas as pd

import streamlit as st


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    st.write("COLUMNS:", list(df.columns))
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    #df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["sales"])
    return df


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df.groupby("month", as_index=False)["sales"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )
    m["prev_sales"] = m["sales"].shift(1)
    m["growth_pct"] = ((m["sales"] - m["prev_sales"]) / m["prev_sales"]) * 100
    return m


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    r = df.groupby("region", as_index=False)["sales"].sum()
    r = r.sort_values("sales", ascending=False).reset_index(drop=True)
    return r


def detect_anomalies(month_df: pd.DataFrame, threshold_pct: float = 10.0):
    alerts = []
    for i in range(1, len(month_df)):
        cur = month_df.loc[i, "sales"]
        prev = month_df.loc[i, "prev_sales"]
        month = month_df.loc[i, "month"]

        if prev and prev != 0:
            change = ((cur - prev) / prev) * 100
            if abs(change) >= threshold_pct:
                direction = "spike" if change > 0 else "drop"
                alerts.append((month, direction, round(change, 2)))

    return alerts


def generate_insights(df: pd.DataFrame) -> list[str]:
    df = prepare_data(df)

    if df.empty:
        return ["No valid rows found. Please check your CSV format."]

    m = monthly_summary(df)
    r = region_summary(df)

    total_sales = int(df["sales"].sum())
    avg_monthly = int(m["sales"].mean())

    insights = []
    insights.append(f"Total sales across the dataset: {total_sales:,}.")
    insights.append(f"Average monthly sales: {avg_monthly:,}.")

    if len(m) >= 2:
        last = m.iloc[-1]
        if pd.notna(last["growth_pct"]):
            gp = round(float(last["growth_pct"]), 2)
            if gp >= 0:
                insights.append(f"Latest month ({last['month']}) grew by {gp}% compared to previous month.")
            else:
                insights.append(f"Latest month ({last['month']}) dropped by {abs(gp)}% compared to previous month.")

    if not r.empty:
        top = r.iloc[0]
        worst = r.iloc[-1]
        insights.append(f"Top performing region: {top['region']} ({int(top['sales']):,}).")
        insights.append(f"Worst performing region: {worst['region']} ({int(worst['sales']):,}).")

    alerts = detect_anomalies(m, threshold_pct=10.0)
    if alerts:
        for month, direction, pct in alerts[-3:]:
            insights.append(f"Detected a {direction} in {month}: {pct}% change vs previous month.")
    else:
        insights.append("No major spikes/drops detected based on the current threshold.")

    # Explanation-style sentence (simple and recruiter-friendly)
    if len(m) >= 2:
        last_month = m.iloc[-1]["month"]
        worst_region = r.iloc[-1]["region"]
        insights.append(
            f"If performance changed recently, check whether {worst_region} contributed heavily during {last_month}."
        )

    return insights"""

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def _find_best_date_column(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    if not cols:
        return None

    name_hints = ["date", "time", "timestamp", "month", "day", "created", "order", "invoice"]
    preferred = [c for c in cols if any(h in c for h in name_hints)]
    candidates = preferred if preferred else cols

    best_col, best_score = None, 0.0
    for c in candidates:
        parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        score = float(parsed.notna().mean())  # parseable ratio
        if score > best_score:
            best_score = score
            best_col = c

    return best_col if best_score >= 0.5 else None


def _find_best_numeric_column(df: pd.DataFrame) -> str | None:
    sales_hints = ["sales", "revenue", "amount", "value", "price", "total", "profit"]
    hinted = [c for c in df.columns if any(h in c for h in sales_hints)]
    if hinted:
        return hinted[0]

    best_col, best_score = None, -1
    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        score = int(converted.notna().sum())
        if score > best_score:
            best_score = score
            best_col = c

    return best_col if best_score > 0 else None


def _find_best_region_column(df: pd.DataFrame) -> str | None:
    region_hints = ["region", "state", "city", "segment", "category", "area", "zone"]
    hinted = [c for c in df.columns if any(h in c for h in region_hints)]
    return hinted[0] if hinted else None


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)

    date_col = _find_best_date_column(df)
    if date_col is None:
        raise ValueError(
            "Couldn't auto-detect a date column. "
            "Add a column like date/month/timestamp or ensure values look like dates."
        )

    num_col = _find_best_numeric_column(df)
    if num_col is None:
        raise ValueError(
            "Couldn't auto-detect a numeric sales/value column. "
            "Add a column like sales/revenue/amount or ensure it contains numbers."
        )

    region_col = _find_best_region_column(df)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    out["sales"] = pd.to_numeric(df[num_col], errors="coerce")

    if region_col:
        out["region"] = df[region_col].astype(str).str.strip()
    else:
        out["region"] = "All"

    out = out.dropna(subset=["date", "sales"])
    out["month"] = out["date"].dt.to_period("M").astype(str)
    return out


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby("month", as_index=False)["sales"].sum().sort_values("month").reset_index(drop=True)
    m["prev_sales"] = m["sales"].shift(1)
    m["growth_pct"] = ((m["sales"] - m["prev_sales"]) / m["prev_sales"]) * 100
    return m


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    r = df.groupby("region", as_index=False)["sales"].sum().sort_values("sales", ascending=False).reset_index(drop=True)
    return r


def detect_anomalies(month_df: pd.DataFrame, threshold_pct: float = 10.0) -> list[tuple[str, str, float]]:
    alerts = []
    for i in range(1, len(month_df)):
        cur = float(month_df.loc[i, "sales"])
        prev = month_df.loc[i, "prev_sales"]
        month = str(month_df.loc[i, "month"])

        if pd.isna(prev) or float(prev) == 0:
            continue

        change = ((cur - float(prev)) / float(prev)) * 100
        if abs(change) >= threshold_pct:
            alerts.append((month, "spike" if change > 0 else "drop", round(float(change), 2)))
    return alerts


def generate_insights(raw_df: pd.DataFrame) -> list[str]:
    df = prepare_data(raw_df)

    if df.empty:
        return ["No usable rows after cleaning. Please check your CSV values."]

    m = monthly_summary(df)
    r = region_summary(df)

    total_sales = float(df["sales"].sum())
    avg_monthly = float(m["sales"].mean())

    insights = [
        f"Total value across dataset: {total_sales:,.0f}.",
        f"Average monthly value: {avg_monthly:,.0f}.",
    ]

    if len(m) >= 2 and pd.notna(m.iloc[-1]["growth_pct"]):
        last = m.iloc[-1]
        gp = float(last["growth_pct"])
        if gp >= 0:
            insights.append(f"Latest month ({last['month']}) grew by {gp:.2f}% vs previous month.")
        else:
            insights.append(f"Latest month ({last['month']}) dropped by {abs(gp):.2f}% vs previous month.")

    if not r.empty:
        top = r.iloc[0]
        worst = r.iloc[-1]
        insights.append(f"Top segment/region: {top['region']} ({float(top['sales']):,.0f}).")
        insights.append(f"Lowest segment/region: {worst['region']} ({float(worst['sales']):,.0f}).")

    alerts = detect_anomalies(m, threshold_pct=10.0)
    if alerts:
        for month, direction, pct in alerts[-3:]:
            insights.append(f"Detected a {direction} in {month}: {pct:.2f}% change vs previous month.")
    else:
        insights.append("No major spikes/drops detected (threshold: 10%).")

    # Simple human explanation
    if not r.empty and len(m) >= 1:
        insights.append(
            f"If the latest change needs investigation, start by checking '{r.iloc[-1]['region']}' for the latest month ({m.iloc[-1]['month']})."
        )

    return insights
