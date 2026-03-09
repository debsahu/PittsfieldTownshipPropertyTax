"""Core analysis computations for property tax assessment analysis."""

import pandas as pd
import numpy as np


# Minimum sale price to filter out lot-only sales
LOT_ONLY_THRESHOLD = 150_000


def get_ecf_trend(data: dict, area_code: str) -> dict[int, float | None]:
    """Get the area-level ECF for each year from ECF Summaries."""
    trend = {}
    for year, df in data["ecf_summaries"].items():
        match = df[df["ECF_Area"] == area_code]
        if len(match) > 0 and pd.notna(match.iloc[0]["Ave_ECF"]):
            trend[year] = float(match.iloc[0]["Ave_ECF"])
        else:
            trend[year] = None
    return trend


def get_ecf_properties(data: dict, area_code: str) -> list[dict]:
    """Get individual property ECF values for the area across all years."""
    rows = []
    for year, df in data["ecf_analysis"].items():
        area_col = "ECF_Area_Code" if "ECF_Area_Code" in df.columns else "ECF_Area"
        if area_col not in df.columns:
            continue
        area_df = df[df[area_col] == area_code].copy()
        area_df = area_df[area_df["ECF"].notna()]
        for _, row in area_df.iterrows():
            rows.append({
                "year": year,
                "address": str(row.get("Street_Address", "")).strip(),
                "parcel": str(row.get("Parcel_Number", "")).strip(),
                "sale_price": row.get("Sale_Price"),
                "cost_man": row.get("Cost_Man"),
                "ecf": row.get("ECF"),
            })
    return rows


def get_subdivision_name(data: dict, area_code: str) -> str:
    """Get the subdivision name for an area code."""
    for year in [2026, 2025, 2024]:
        if year in data["ecf_summaries"]:
            match = data["ecf_summaries"][year]
            match = match[match["ECF_Area"] == area_code]
            if len(match) > 0:
                name = str(match.iloc[0].get("Subdivision", "")).strip()
                if name and name != "nan":
                    return name
    # Try land adjustments
    for year in [2026, 2025, 2024]:
        if year in data["land_adj"]:
            match = data["land_adj"][year]
            match = match[match["Area_Code"] == area_code]
            if len(match) > 0:
                name = str(match.iloc[0].get("Subdivision", "")).strip()
                if name and name != "nan":
                    return name
    return area_code


def get_comparable_sales(data: dict, area_code: str) -> pd.DataFrame:
    """Get arm's-length completed-home sales for the area across all years."""
    all_sales = []
    for year, df in data["sales"].items():
        if "ECF_Area" not in df.columns:
            continue
        area_df = df[df["ECF_Area"] == area_code].copy()
        # Filter arm's-length sales
        if "Terms_of_Sale" in area_df.columns:
            area_df = area_df[area_df["Terms_of_Sale"].str.contains("ARM", na=False, case=False)]
        # Filter out lot-only sales
        area_df = area_df[area_df["Adj_Sale"].notna() & (area_df["Adj_Sale"] >= LOT_ONLY_THRESHOLD)]
        if len(area_df) > 0:
            area_df = area_df[["Parcel_Number", "Street_Address", "Sale_Date",
                               "Sale_Price", "Adj_Sale", "_year"]].copy()
            all_sales.append(area_df)

    if not all_sales:
        return pd.DataFrame()

    result = pd.concat(all_sales, ignore_index=True)
    # Remove duplicate sales (same parcel + same sale date across years)
    # Normalize dates to avoid format mismatches (05/25/2022 vs 5/25/2022)
    result["_norm_date"] = pd.to_datetime(result["Sale_Date"], format="mixed", dayfirst=False).dt.strftime("%Y-%m-%d")
    result["_dedup_key"] = result["Parcel_Number"].str.strip() + "_" + result["_norm_date"]
    result = result.drop_duplicates(subset="_dedup_key", keep="last").drop(columns=["_dedup_key", "_norm_date"])
    result = result.sort_values("Sale_Date", ascending=False)
    return result


def compute_sales_stats(sales_df: pd.DataFrame, user_tcv: float) -> dict:
    """Compute summary statistics comparing sales to user's TCV."""
    if sales_df.empty:
        return {"count": 0}

    prices = sales_df["Adj_Sale"].dropna()
    count = len(prices)
    mean = float(prices.mean())
    median = float(prices.median())
    min_price = float(prices.min())
    max_price = float(prices.max())
    pct_below_tcv = float((prices < user_tcv).sum() / count * 100) if count > 0 else 0

    return {
        "count": count,
        "mean": mean,
        "median": median,
        "min": min_price,
        "max": max_price,
        "pct_below_tcv": pct_below_tcv,
        "pct_above_tcv": 100 - pct_below_tcv,
        "delta_from_median": user_tcv - median,
        "delta_pct": (user_tcv - median) / median * 100 if median > 0 else 0,
    }


def get_land_value_trends(data: dict, area_code: str) -> list[dict]:
    """Get land adjustment factors and values per year."""
    rows = []
    for year in [2024, 2025, 2026]:
        if year not in data["land_adj"]:
            continue
        df = data["land_adj"][year]
        match = df[df["Area_Code"] == area_code]
        if len(match) == 0:
            continue
        factor = match.iloc[0].get("Adjust_Factor")

        # Get land values from land analysis (use Area_Code, not ECF_Area which
        # may have concatenated values in 2026 like "AR-4AR4-MEADOWS")
        prior_lv = current_lv = None
        if year in data["land"]:
            land_df = data["land"][year]
            # Prefer Area_Code column (clean), fall back to ECF_Area
            if "Area_Code" in land_df.columns:
                land_area = land_df[land_df["Area_Code"].str.strip() == area_code]
            else:
                land_area = land_df[land_df.get("ECF_Area", pd.Series(dtype=str)).str.strip() == area_code]
            if len(land_area) > 0:
                if "Land_Value_Prior" in land_area.columns:
                    vals = land_area["Land_Value_Prior"].dropna()
                    if len(vals) > 0:
                        prior_lv = float(vals.mode().iloc[0]) if len(vals.mode()) > 0 else float(vals.median())
                if "Land_Value_Current" in land_area.columns:
                    vals = land_area["Land_Value_Current"].dropna()
                    if len(vals) > 0:
                        current_lv = float(vals.mode().iloc[0]) if len(vals.mode()) > 0 else float(vals.median())

        rows.append({
            "year": year,
            "adjust_factor": float(factor) if pd.notna(factor) else None,
            "prior_lv": prior_lv,
            "current_lv": current_lv,
        })
    return rows


def check_sales_coverage(data: dict, area_code: str) -> dict[int, int]:
    """Count how many sales for this area appear in each year's sales study."""
    coverage = {}
    for year, df in data["sales"].items():
        if "ECF_Area" not in df.columns:
            coverage[year] = 0
            continue
        count = (df["ECF_Area"] == area_code).sum()
        coverage[year] = int(count)
    return coverage


def compute_ecf_adjusted_value(ecf: float | None, user_tcv: float) -> float | None:
    """Compute ECF-adjusted TCV."""
    if ecf is None or ecf <= 0:
        return None
    return user_tcv * ecf


def get_overvaluation_pct(ecf: float | None) -> float | None:
    """Get the overvaluation percentage from ECF."""
    if ecf is None or ecf >= 1.0:
        return None
    return (1 - ecf) * 100


def _compute_recommended_values(ecf_2026: float | None, user_tcv: float,
                                 sales_stats: dict,
                                 prop=None) -> tuple[float, float]:
    """Compute recommended TCV range (low, high) for the appeal.

    When a Record Card is available with cost-approach data and ECF < 1.0,
    the primary recommendation is the ECF-adjusted cost-approach TCV, which
    is property-specific (accounts for sqft, condition, age, etc.).

    Otherwise falls back to area-level median of comparable sales.
    """
    candidates = []
    if ecf_2026 is not None and ecf_2026 < 1.0:
        candidates.append(user_tcv * ecf_2026)
    if sales_stats.get("count", 0) > 0:
        candidates.append(sales_stats["median"])
        candidates.append(sales_stats["mean"])
    if not candidates:
        return user_tcv, user_tcv

    # Property-specific ECF-adjusted cost approach value
    # (cost approach already scales by sqft, condition, age, style)
    prop_ecf_tcv = None
    if (prop and prop.total_depr_cost and prop.total_depr_cost > 0
            and ecf_2026 is not None and ecf_2026 < 1.0):
        prop_ecf_tcv = prop.total_depr_cost * ecf_2026

    has_sales = sales_stats.get("count", 0) > 0
    median = sales_stats.get("median", 0) if has_sales else 0

    if has_sales and prop_ecf_tcv:
        if median >= user_tcv:
            # Sales median at/above assessed TCV -- no basis for appeal
            primary = median
        else:
            # Both prop_ecf_tcv and sales median are below TCV.
            # Average them for a property-specific yet moderate value.
            # prop_ecf_tcv varies by property (sqft, condition, age),
            # while sales median anchors to market.
            primary = (prop_ecf_tcv + median) / 2
    elif has_sales:
        primary = median
    elif prop_ecf_tcv:
        primary = prop_ecf_tcv
    elif ecf_2026 is not None and ecf_2026 < 1.0:
        primary = user_tcv * ecf_2026
    else:
        primary = user_tcv
    return primary, max(candidates)


def generate_appeal_summary(area_code: str, subdivision: str, user_sev: float,
                            ecf_trend: dict, sales_stats: dict,
                            land_trends: list, coverage: dict,
                            ecf_properties: list,
                            sales_df: pd.DataFrame | None = None,
                            prop=None) -> str:
    """Generate a comprehensive L4035-style appeal petition.

    Args:
        prop: Optional PropertyData from RC.pdf parser (for property details).
        sales_df: Optional DataFrame of comparable sales.
    """
    user_tcv = user_sev * 2
    ecf_2026 = ecf_trend.get(2026)
    ecf_adjusted = user_tcv * ecf_2026 if ecf_2026 and ecf_2026 < 1.0 else None
    rec_low, rec_high = _compute_recommended_values(ecf_2026, user_tcv, sales_stats, prop=prop)
    rec_sev = round(rec_low / 2 / 5000) * 5000  # Round to nearest $5,000

    # If recommended value >= current assessment, no basis for appeal
    no_appeal = rec_sev >= user_sev

    # Property-specific ECF-adjusted cost-approach value
    prop_ecf_tcv = None
    if (prop and prop.total_depr_cost and prop.total_depr_cost > 0
            and ecf_2026 is not None and ecf_2026 < 1.0):
        prop_ecf_tcv = prop.total_depr_cost * ecf_2026

    L = []  # lines

    if no_appeal:
        L.append("=" * 70)
        L.append("APPEAL ANALYSIS -- NOT RECOMMENDED")
        L.append("=" * 70)
        L.append("")
        if prop and prop.address:
            L.append(f"Property:  {prop.address}")
        L.append(f"Area:      {area_code} ({subdivision})")
        L.append(f"Your SEV:  ${user_sev:,.0f}  (TCV: ${user_tcv:,.0f})")
        L.append("")
        L.append("Based on the available evidence, an appeal is NOT recommended")
        L.append("for this property. The current assessment appears to be at or")
        L.append("below market value:")
        L.append("")
        if sales_stats.get("count", 0) > 0:
            L.append(f"  - Median of {sales_stats['count']} comparable sales: ${sales_stats['median']:,.0f}")
            L.append(f"  - Average of {sales_stats['count']} comparable sales: ${sales_stats['mean']:,.0f}")
        if ecf_adjusted:
            L.append(f"  - ECF-adjusted value (ECF={ecf_2026:.3f}):   ${ecf_adjusted:,.0f}")
        if prop_ecf_tcv:
            L.append(f"  - Property-specific ECF-adjusted TCV:    ${prop_ecf_tcv:,.0f}")
        L.append(f"  - Your current TCV:                      ${user_tcv:,.0f}")
        L.append("")
        if sales_stats.get("count", 0) > 0 and sales_stats["median"] >= user_tcv:
            L.append(f"The median sale price (${sales_stats['median']:,.0f}) is at or above your")
            L.append(f"TCV (${user_tcv:,.0f}), meaning comparable sales support the current")
            L.append(f"assessment. The Board of Review is unlikely to grant a reduction.")
        elif ecf_adjusted and ecf_adjusted < user_tcv and sales_stats.get("count", 0) > 0:
            L.append(f"While the ECF ({ecf_2026:.3f}) suggests some overvaluation, the median")
            L.append(f"sale price (${sales_stats['median']:,.0f}) supports the current assessment.")
            L.append(f"An ECF-only argument would be aggressive and may not succeed.")
        L.append("")
        L.append("=" * 70)
        return "\n".join(L)

    rec_tcv = rec_sev * 2

    # Taxable value (from RC.pdf if available)
    taxable = None
    if prop and prop.assessment_history:
        h2026 = next((h for h in prop.assessment_history if h.year == 2026), None)
        if h2026 and h2026.taxable_value:
            taxable = h2026.taxable_value

    # ============================================================
    # LETTER HEADER -- addressed to Board of Review
    # ============================================================
    L.append("                                          Date: ___/___/2026")
    L.append("")
    L.append("Board of Review")
    L.append("Pittsfield Charter Township")
    L.append("6201 W Michigan Avenue")
    L.append("Ann Arbor, MI 48108-9721")
    L.append("")
    L.append("RE: PETITION TO THE BOARD OF REVIEW")
    L.append(f"    Tax Year 2026 -- Residential Property (Classification 401)")
    if prop and prop.parcel_number:
        L.append(f"    Parcel: {prop.parcel_number}")
    if prop and prop.address:
        L.append(f"    Property: {prop.address}, Ypsilanti, MI 48197")
    L.append(f"    ECF Area: {area_code} ({subdivision})")
    L.append("")
    L.append("Dear Members of the Board of Review:")
    L.append("")
    L.append("I respectfully submit this written petition pursuant to MCL 211.30")
    L.append("and the General Property Tax Act, requesting that the Board consider")
    L.append("this petition in lieu of a personal appearance per MCL 211.30(4).")
    L.append("")

    # ============================================================
    # PROPERTY INFORMATION
    # ============================================================
    L.append("=" * 65)
    L.append("PROPERTY INFORMATION")
    L.append("=" * 65)
    if prop:
        if prop.parcel_number:
            L.append(f"Parcel Number:    {prop.parcel_number}")
        if prop.address:
            L.append(f"Property Address: {prop.address}, Ypsilanti, MI 48197")
    L.append(f"Township:         Pittsfield Charter Township")
    L.append(f"County:           Washtenaw")
    L.append(f"School District:  Ann Arbor Public Schools")
    L.append(f"Classification:   Residential (401)")
    L.append(f"ECF Area:         {area_code} ({subdivision})")
    if prop:
        details = []
        if prop.style:
            details.append(prop.style)
        if prop.year_built:
            details.append(f"Built {prop.year_built}")
        if prop.floor_area:
            details.append(f"{prop.floor_area:,} SF")
        if prop.condition:
            details.append(f"Condition: {prop.condition}")
        if details:
            L.append(f"Property:         {', '.join(details)}")
    L.append("")

    # ============================================================
    # ASSESSMENT VALUES TABLE
    # ============================================================
    L.append("=" * 65)
    L.append("ASSESSMENT VALUES")
    L.append("=" * 65)
    L.append(f"{'':28s} {'Current':>12s}  {'Petitioner':>12s}  {'Difference':>12s}")
    L.append(f"{'-' * 28} {'-' * 12}  {'-' * 12}  {'-' * 12}")
    L.append(f"{'Assessed Value (SEV)':28s} ${user_sev:>11,.0f}  ${rec_sev:>11,.0f}  ${rec_sev - user_sev:>+11,.0f}")
    L.append(f"{'True Cash Value (TCV)':28s} ${user_tcv:>11,.0f}  ${rec_tcv:>11,.0f}  ${rec_tcv - user_tcv:>+11,.0f}")
    if taxable:
        L.append(f"{'Taxable Value':28s} ${taxable:>11,.0f}  ${taxable:>11,.0f}  ${'0':>11s}")
    L.append("")

    # Show how the recommended value was derived
    L.append("Basis for petitioner's value:")
    if prop_ecf_tcv:
        L.append(f"  Assessor's cost-approach TCV for this property: ${prop.total_depr_cost:,.0f}")
        L.append(f"  Township ECF for {area_code} (2026):              {ecf_2026:.3f}")
        L.append(f"  ECF-adjusted TCV ({prop.total_depr_cost:,.0f} x {ecf_2026:.3f}):  ${prop_ecf_tcv:,.0f}")
        if prop.floor_area:
            ppsf = prop_ecf_tcv / prop.floor_area
            L.append(f"  Implied $/SF ({prop.floor_area:,} SF):                ${ppsf:,.0f}/SF")
        L.append(f"  Recommended SEV (rounded to $5,000):      ${rec_sev:,.0f}")
    elif sales_stats.get("count", 0) > 0:
        L.append(f"  Median of {sales_stats['count']} comparable sales:  ${sales_stats['median']:,.0f}")
        L.append(f"  Recommended SEV (rounded to $5,000):    ${rec_sev:,.0f}")
    L.append("")

    # ============================================================
    # GROUNDS FOR APPEAL
    # ============================================================
    L.append("=" * 65)
    L.append("GROUNDS FOR APPEAL")
    L.append("=" * 65)
    L.append(f"The petitioner contends that the 2026 assessed value of")
    L.append(f"${user_sev:,.0f} (implying a TCV of ${user_tcv:,.0f}) exceeds the usual")
    L.append(f"selling price for comparable properties in the {subdivision}")
    L.append(f"subdivision ({area_code}) and surrounding area.")
    L.append("")
    L.append(f"The petitioner requests a reduction to an assessed value of")
    L.append(f"${rec_sev:,.0f} (TCV of ${rec_tcv:,.0f}), supported by the following")
    L.append(f"evidence from the township's own records and market data.")
    L.append("")

    evidence_num = 1

    # ============================================================
    # EVIDENCE 1: ECF DATA
    # ============================================================
    if ecf_2026 is not None:
        L.append("-" * 65)
        L.append(f"EVIDENCE {evidence_num}: TOWNSHIP ECF DATA " +
                 ("CONFIRMS OVER-ASSESSMENT" if ecf_2026 < 1.0 else "ANALYSIS"))
        L.append("-" * 65)

        if ecf_2026 < 1.0:
            overval = (1 - ecf_2026) * 100
            L.append(f"The township's own Economic Condition Factor (ECF) for {area_code}")
            L.append(f"is {ecf_2026:.3f} (2026), meaning the cost-approach valuations used")
            L.append(f"by the assessor EXCEED actual market sale prices by approx.")
            L.append(f"{overval:.1f}%. This pattern is consistent across years:")
            L.append("")
            for year in sorted(ecf_trend.keys()):
                val = ecf_trend[year]
                if val is not None:
                    yr_overval = (1 - val) * 100
                    L.append(f"  {year} ECF: {val:.3f} (cost exceeds market by {yr_overval:.1f}%)")
                else:
                    L.append(f"  {year} ECF: Not available (area not in study)")
            L.append("")
            L.append(f"Applying the ECF to the current assessment:")
            L.append(f"  TCV x ECF = ECF-Adjusted TCV")
            L.append(f"  ${user_tcv:,.0f} x {ecf_2026:.3f} = ${ecf_adjusted:,.0f}")
            L.append(f"  Implied over-assessment: ${user_tcv - (ecf_adjusted or 0):,.0f}")
            if prop_ecf_tcv:
                L.append("")
                L.append(f"Using the assessor's own cost-approach value for THIS property:")
                L.append(f"  Cost TCV x ECF = Property-specific market TCV")
                L.append(f"  ${prop.total_depr_cost:,.0f} x {ecf_2026:.3f} = ${prop_ecf_tcv:,.0f}")
                if prop.floor_area:
                    L.append(f"  (Based on {prop.floor_area:,} SF of floor area)")
        else:
            L.append(f"The ECF for {area_code} is {ecf_2026:.3f} (at or above 1.0),")
            L.append(f"indicating the cost approach does not systematically overvalue")
            L.append(f"properties in this area.")
        L.append("")
        evidence_num += 1

    # ============================================================
    # EVIDENCE 2: COMPARABLE SALES
    # ============================================================
    if sales_stats.get("count", 0) > 0:
        L.append("-" * 65)
        L.append(f"EVIDENCE {evidence_num}: COMPARABLE SALES ANALYSIS")
        L.append("-" * 65)

        count = sales_stats["count"]
        median = sales_stats["median"]
        mean = sales_stats["mean"]
        below_count = int(count * sales_stats["pct_below_tcv"] / 100)

        if sales_stats["delta_from_median"] > 0:
            L.append(f"The assessed TCV of ${user_tcv:,.0f} exceeds the median sale")
            L.append(f"price of comparable arm's-length sales in {area_code}. Of")
            L.append(f"{count} sales identified across 2024-2026 data,")
            L.append(f"{below_count} ({sales_stats['pct_below_tcv']:.0f}%) sold BELOW the assessed TCV.")
        else:
            L.append(f"Analysis of {count} arm's-length sales in {area_code} from the")
            L.append(f"2024-2026 assessment data:")
        L.append("")

        # Build per-property ECF lookup: address -> {year: ecf}
        ecf_lookup: dict[str, dict[int, float]] = {}
        if ecf_properties:
            for ep in ecf_properties:
                addr_key = str(ep.get("address", "")).strip().upper()
                if addr_key not in ecf_lookup:
                    ecf_lookup[addr_key] = {}
                if pd.notna(ep.get("ecf")):
                    ecf_lookup[addr_key][ep["year"]] = ep["ecf"]

        # Individual sales table
        if sales_df is not None and not sales_df.empty:
            L.append(f"  {'Address':<24s} {'Sale Price':>11s} {'Date':>11s} {'vs TCV':>11s} {'ECF':>6s}")
            L.append(f"  {'-' * 24} {'-' * 11} {'-' * 11} {'-' * 11} {'-' * 6}")
            for _, row in sales_df.iterrows():
                addr = str(row.get("Street_Address", "")).strip()
                addr_display = addr[:24]
                addr_key = addr.upper()
                price = row.get("Adj_Sale")
                date = str(row.get("Sale_Date", "")).strip()[:11]
                if pd.notna(price):
                    diff = price - user_tcv
                    diff_str = f"${diff:+,.0f}"
                    # Look up per-property ECFs (prefer 2026, then 2025, then 2024)
                    prop_ecfs = ecf_lookup.get(addr_key, {})
                    ecf_val = prop_ecfs.get(2026, prop_ecfs.get(2025, prop_ecfs.get(2024)))
                    ecf_str = f"{ecf_val:.3f}" if ecf_val else "  -  "
                    L.append(f"  {addr_display:<24s} ${price:>10,.0f} {date:>11s} {diff_str:>11s} {ecf_str:>6s}")
            L.append("")

        L.append(f"  Summary Statistics:")
        L.append(f"  - Number of sales:       {count}")
        L.append(f"  - Median sale price:     ${median:,.0f}")
        L.append(f"  - Average sale price:    ${mean:,.0f}")
        L.append(f"  - Price range:           ${sales_stats['min']:,.0f} - ${sales_stats['max']:,.0f}")
        if sales_stats["delta_from_median"] > 0:
            L.append(f"  - Your TCV vs median:    +${sales_stats['delta_from_median']:,.0f} ({sales_stats['delta_pct']:.1f}% above)")
            L.append(f"  - Sales below your TCV:  {below_count} of {count} ({sales_stats['pct_below_tcv']:.0f}%)")
        L.append("")
        evidence_num += 1

    # ============================================================
    # EVIDENCE 3: LAND VALUE TRENDS
    # ============================================================
    if land_trends:
        L.append("-" * 65)
        L.append(f"EVIDENCE {evidence_num}: LAND VALUE TREND")
        L.append("-" * 65)
        L.append(f"The {area_code} land values changed as follows over 3 years:")
        L.append("")

        first_lv = None
        last_lv = None
        for lt in land_trends:
            factor = lt.get("adjust_factor")
            current = lt.get("current_lv")
            prior = lt.get("prior_lv")
            if prior and first_lv is None:
                first_lv = prior
            if current:
                last_lv = current
            if factor is not None:
                pct = (factor - 1) * 100
                current_str = f"${current:,.0f}" if current else "N/A"
                L.append(f"  {lt['year']}: {current_str} (Factor: {factor:.4f}, {pct:+.1f}%)")
            elif current:
                L.append(f"  {lt['year']}: ${current:,.0f}")

        if first_lv and last_lv and first_lv > 0:
            total_pct = (last_lv - first_lv) / first_lv * 100
            L.append("")
            L.append(f"  Total change: ${first_lv:,.0f} -> ${last_lv:,.0f} ({total_pct:+.1f}%)")

        if ecf_2026 and ecf_2026 < 1.0:
            L.append("")
            L.append(f"  While land values have increased, the ECF data shows that")
            L.append(f"  building cost valuations consistently exceed market.")
        L.append("")
        evidence_num += 1

    # ============================================================
    # CONCLUSION AND REQUESTED RELIEF
    # ============================================================
    L.append("=" * 65)
    L.append("CONCLUSION AND REQUESTED RELIEF")
    L.append("=" * 65)

    arguments = []
    if ecf_2026 and ecf_2026 < 1.0:
        arguments.append(f"the township's own ECF analysis (ECF = {ecf_2026:.3f})")
    if prop_ecf_tcv:
        arguments.append(f"the property-specific ECF-adjusted cost approach "
                        f"(${prop_ecf_tcv:,.0f})")
    if sales_stats.get("count", 0) > 0 and sales_stats["delta_from_median"] > 0:
        arguments.append(f"comparable sales (median ${sales_stats['median']:,.0f}, "
                        f"mean ${sales_stats['mean']:,.0f})")

    L.append(f"Based on {', '.join(arguments) if arguments else 'the evidence above'},")
    L.append(f"the petitioner respectfully requests that the Board of Review")
    L.append(f"reduce the 2026 assessed value from ${user_sev:,.0f} to")
    L.append(f"${rec_sev:,.0f}, reflecting a True Cash Value of ${rec_tcv:,.0f}.")
    L.append("")

    L.append(f"This value is consistent with:")
    if prop_ecf_tcv:
        L.append(f"  - The assessor's own cost approach for this property,")
        L.append(f"    adjusted by the township ECF (${prop_ecf_tcv:,.0f})")
    if sales_stats.get("count", 0) > 0:
        L.append(f"  - {sales_stats['count']} comparable arm's-length sales")
        L.append(f"    (median ${sales_stats['median']:,.0f}, mean ${sales_stats['mean']:,.0f})")
    if ecf_adjusted and not prop_ecf_tcv:
        L.append(f"  - The township's ECF-adjusted cost approach (${ecf_adjusted:,.0f})")
    L.append(f"  - The sales-comparison approach, the most persuasive valuation")
    L.append(f"    method for residential property under Michigan law")
    L.append(f"    (Meadowlanes Ltd v Holland, 437 Mich 473)")
    L.append("")

    # Legal citations
    L.append("-" * 65)
    L.append("LEGAL BASIS")
    L.append("-" * 65)
    L.append(f"Under MCL 211.27(1), true cash value means the usual selling")
    L.append(f"price. There is no presumption of validity for the assessor's")
    L.append(f"value (Alhi Development Co v Orion Twp, 110 Mich App 764).")
    L.append(f"The sales-comparison approach is the most persuasive valuation")
    L.append(f"method for residential property (Meadowlanes Ltd v Holland,")
    L.append(f"437 Mich 473).")
    L.append("")

    # ============================================================
    # SIGNATURE BLOCK
    # ============================================================
    L.append("Respectfully submitted,")
    L.append("")
    L.append("")
    L.append("Signature: ____________________________")
    L.append("")
    if prop and prop.address:
        L.append(f"Printed Name: _________________________")
        L.append(f"Address: {prop.address}, Ypsilanti, MI 48197")
    else:
        L.append(f"Printed Name: _________________________")
        L.append(f"Address: _____________________________")
    L.append(f"Phone: ________________________________")
    L.append(f"Email: ________________________________")
    L.append("")

    # Footer
    L.append("-" * 65)
    L.append(f"Data source: Pittsfield Township official assessment data")
    L.append(f"(2024-2026), pittsfield-mi.gov/2230/Property-Assessment-Data")
    L.append("")
    L.append(f"APPEAL DEADLINE: March 10, 2026 at 5:00 PM")
    L.append(f"Phone: 734-822-3115 | Email: assessing@pittsfield-mi.gov")

    return "\n".join(L)
