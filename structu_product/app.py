from __future__ import annotations

import math
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover - the app falls back to linear interpolation.
    PchipInterpolator = None


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR.parent / "data.csv"
NORMAL = NormalDist()

REQUIRED_COLUMNS = {"tenor", "atm_vol", "rr25", "bf25", "rr10", "bf10"}
NUMERIC_COLUMNS = ["atm_vol", "rr25", "bf25", "rr10", "bf10"]

BUCKETS = [
    {"bucket": "10D Put", "axis": 0, "delta": -0.10, "quote_col": "vol_10d_put", "n_d1": 0.90},
    {"bucket": "25D Put", "axis": 1, "delta": -0.25, "quote_col": "vol_25d_put", "n_d1": 0.75},
    {"bucket": "ATM", "axis": 2, "delta": 0.00, "quote_col": "atm_vol", "n_d1": None},
    {"bucket": "25D Call", "axis": 3, "delta": 0.25, "quote_col": "vol_25d_call", "n_d1": 0.25},
    {"bucket": "10D Call", "axis": 4, "delta": 0.10, "quote_col": "vol_10d_call", "n_d1": 0.10},
]
BUCKET_LABELS = [bucket["bucket"] for bucket in BUCKETS]


@dataclass(frozen=True)
class Leg:
    name: str
    option_type: str
    quantity: float
    strike: float


@dataclass
class StructureResult:
    name: str
    legs: list[Leg]
    premium_per_eur: float
    premium_usd: float
    premium_eur: float
    premium_pct: float
    protection: str
    upside: str
    tail_risk: str
    comment: str
    score: float
    leg_rows: list[dict]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #070b0f;
            color: #e7edf3;
        }
        [data-testid="stSidebar"] {
            background: #0a1017;
            border-right: 1px solid #1d2a36;
        }
        .block-container {
            max-width: 1540px;
            padding-top: 1.1rem;
            padding-bottom: 2.2rem;
        }
        h1, h2, h3 {
            color: #f2f6fa;
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: #0d151d;
            border: 1px solid #1d2b38;
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
        }
        div[data-testid="stMetricValue"] {
            color: #f6b642;
            font-size: 1.22rem;
        }
        div[data-testid="stMetricLabel"] {
            color: #aeb9c4;
        }
        .bbg-header {
            border: 1px solid #1d2b38;
            border-radius: 8px;
            background: linear-gradient(90deg, #101923 0%, #0c1219 58%, #121407 100%);
            padding: 0.85rem 1rem;
            margin-bottom: 0.9rem;
        }
        .bbg-title {
            font-size: 1.65rem;
            font-weight: 740;
            color: #f2f6fa;
        }
        .bbg-subtitle {
            margin-top: 0.22rem;
            color: #aeb9c4;
            font-size: 0.92rem;
        }
        .ticker-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.8rem;
        }
        .ticker-cell {
            border: 1px solid #253444;
            border-radius: 6px;
            padding: 0.38rem 0.58rem;
            background: #080e14;
            color: #d6dee6;
            font-size: 0.83rem;
        }
        .ticker-value {
            color: #f6b642;
            font-weight: 720;
            margin-left: 0.25rem;
        }
        .desk-note {
            border-left: 3px solid #f6b642;
            background: #0c141c;
            padding: 0.75rem 0.85rem;
            border-radius: 6px;
            color: #d6dee6;
            margin: 0.55rem 0 0.85rem 0;
        }
        .recommendation {
            border: 1px solid #2a6f55;
            background: #071711;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            color: #d8f3e6;
        }
        .info-panel {
            border: 1px solid #1d2b38;
            background: #0c141c;
            border-radius: 8px;
            padding: 0.8rem 0.95rem;
            color: #d6dee6;
            margin: 0.55rem 0 0.85rem 0;
        }
        .panel-title {
            color: #f6b642;
            font-weight: 740;
            margin-bottom: 0.25rem;
        }
        .small-muted {
            color: #8d9aaa;
            font-size: 0.86rem;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: #0d151d;
            border: 1px solid #1d2b38;
            border-radius: 7px;
            padding: 0.25rem 0.45rem;
            margin-bottom: 0.25rem;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            border-color: #f6b642;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tenor_to_years(tenor: str) -> float:
    label = str(tenor).strip().upper()
    if len(label) < 2:
        raise ValueError(f"Invalid tenor: {tenor}")
    value = float(label[:-1])
    unit = label[-1]
    if unit == "D":
        return value / 365.0
    if unit == "W":
        return 7.0 * value / 365.0
    if unit == "M":
        return value / 12.0
    if unit == "Y":
        return value
    raise ValueError(f"Invalid tenor unit: {tenor}")


def parse_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("\u202f", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def decode_csv_source(source) -> tuple[pd.DataFrame, str]:
    if source is None:
        raw = DEFAULT_CSV.read_bytes()
        source_name = str(DEFAULT_CSV)
    elif hasattr(source, "getvalue"):
        raw = source.getvalue()
        source_name = source.name
    else:
        source_path = Path(source)
        raw = source_path.read_bytes()
        source_name = str(source_path)

    text = raw.decode("utf-8-sig")
    header = text.splitlines()[0] if text.splitlines() else ""
    sep = ";" if ";" in header else ","
    df = pd.read_csv(StringIO(text), sep=sep, dtype=str)
    return df, source_name


def load_quotes(source, spot: float, rd: float, rf: float) -> tuple[pd.DataFrame, str]:
    df, source_name = decode_csv_source(source)
    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    quotes = df[list(REQUIRED_COLUMNS)].copy()
    quotes["tenor"] = quotes["tenor"].astype(str).str.upper().str.strip()

    for col in NUMERIC_COLUMNS:
        quotes[col] = parse_numeric_series(quotes[col])

    bad_numeric = quotes[quotes[NUMERIC_COLUMNS].isna().any(axis=1)]
    if not bad_numeric.empty:
        raise ValueError(
            "Invalid numeric values in rows: "
            + ", ".join(str(i + 1) for i in bad_numeric.index.tolist())
        )

    try:
        quotes["maturity_years"] = quotes["tenor"].map(tenor_to_years)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    quotes = quotes.sort_values("maturity_years").reset_index(drop=True)
    quotes["forward"] = spot * np.exp((rd - rf) * quotes["maturity_years"])
    quotes["vol_25d_call"] = quotes["atm_vol"] + quotes["bf25"] + quotes["rr25"] / 2.0
    quotes["vol_25d_put"] = quotes["atm_vol"] + quotes["bf25"] - quotes["rr25"] / 2.0
    quotes["vol_10d_call"] = quotes["atm_vol"] + quotes["bf10"] + quotes["rr10"] / 2.0
    quotes["vol_10d_put"] = quotes["atm_vol"] + quotes["bf10"] - quotes["rr10"] / 2.0

    wing_cols = ["vol_10d_put", "vol_25d_put", "atm_vol", "vol_25d_call", "vol_10d_call"]
    if (quotes[wing_cols] <= 0).any().any():
        raise ValueError("Reconstructed smile contains non-positive volatilities.")

    return quotes, source_name


def normal_cdf(x: float) -> float:
    return NORMAL.cdf(x)


def normal_ppf(p: float) -> float:
    return NORMAL.inv_cdf(p)


def forward_rate(spot: float, rd: float, rf: float, maturity_years: float) -> float:
    return spot * math.exp((rd - rf) * maturity_years)


def strike_from_forward_delta(forward: float, vol_pct: float, maturity_years: float, n_d1: float | None) -> float:
    if n_d1 is None:
        return forward
    vol = vol_pct / 100.0
    std = max(vol * math.sqrt(max(maturity_years, 1e-10)), 1e-10)
    return forward * math.exp(0.5 * vol * vol * maturity_years - std * normal_ppf(n_d1))


def build_bucket_surface(quotes: pd.DataFrame, spot: float) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in quotes.iterrows():
        for bucket in BUCKETS:
            vol_pct = float(row[bucket["quote_col"]])
            strike = strike_from_forward_delta(
                forward=float(row["forward"]),
                vol_pct=vol_pct,
                maturity_years=float(row["maturity_years"]),
                n_d1=bucket["n_d1"],
            )
            rows.append(
                {
                    "tenor": row["tenor"],
                    "maturity_years": float(row["maturity_years"]),
                    "forward": float(row["forward"]),
                    "bucket": bucket["bucket"],
                    "bucket_axis": bucket["axis"],
                    "delta_bucket": bucket["delta"],
                    "strike": strike,
                    "strike_over_spot": strike / spot,
                    "log_moneyness": math.log(strike / float(row["forward"])),
                    "vol_pct": vol_pct,
                }
            )
    return pd.DataFrame(rows).sort_values(["maturity_years", "bucket_axis"]).reset_index(drop=True)


def interpolate_1d(x: np.ndarray, y: np.ndarray, x_new: float, method: str) -> float:
    order = np.argsort(x)
    x_sorted = np.asarray(x[order], dtype=float)
    y_sorted = np.asarray(y[order], dtype=float)
    x_clamped = min(max(float(x_new), float(x_sorted.min())), float(x_sorted.max()))

    if method == "PCHIP" and PchipInterpolator is not None and len(x_sorted) >= 3:
        return float(PchipInterpolator(x_sorted, y_sorted, extrapolate=False)(x_clamped))
    return float(np.interp(x_clamped, x_sorted, y_sorted))


def surface_vol_pct(
    bucket_surface: pd.DataFrame,
    spot: float,
    rd: float,
    rf: float,
    maturity_years: float,
    strike: float,
    method: str,
) -> float:
    market_maturities = np.array(sorted(bucket_surface["maturity_years"].unique()), dtype=float)
    total_variances = []

    for market_t in market_maturities:
        group = bucket_surface[bucket_surface["maturity_years"] == market_t].sort_values("log_moneyness")
        market_forward = forward_rate(spot, rd, rf, market_t)
        x = math.log(max(strike, 1e-10) / market_forward)
        vol = interpolate_1d(
            group["log_moneyness"].to_numpy(dtype=float),
            group["vol_pct"].to_numpy(dtype=float),
            x,
            method,
        )
        vol_dec = max(vol, 0.01) / 100.0
        total_variances.append(vol_dec * vol_dec * max(market_t, 1e-10))

    maturity = min(max(maturity_years, market_maturities.min()), market_maturities.max())
    total_var = interpolate_1d(market_maturities, np.asarray(total_variances), maturity, method)
    return math.sqrt(max(total_var, 1e-12) / max(maturity_years, 1e-10)) * 100.0


def fx_option_price(
    spot: float,
    strike: float,
    maturity_years: float,
    rd: float,
    rf: float,
    vol_pct: float,
    option_type: str,
) -> float:
    if maturity_years <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)

    vol = max(vol_pct / 100.0, 1e-8)
    std = max(vol * math.sqrt(maturity_years), 1e-12)
    d1 = (math.log(spot / strike) + (rd - rf + 0.5 * vol * vol) * maturity_years) / std
    d2 = d1 - std

    if option_type == "call":
        return spot * math.exp(-rf * maturity_years) * normal_cdf(d1) - strike * math.exp(-rd * maturity_years) * normal_cdf(d2)
    if option_type == "put":
        return strike * math.exp(-rd * maturity_years) * normal_cdf(-d2) - spot * math.exp(-rf * maturity_years) * normal_cdf(-d1)
    raise ValueError(f"Unknown option type: {option_type}")


def bisection(fn, low: float, high: float, max_iter: int = 80) -> float | None:
    f_low = fn(low)
    f_high = fn(high)
    if not np.isfinite(f_low) or not np.isfinite(f_high) or f_low * f_high > 0:
        return None

    lo, hi = low, high
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = fn(mid)
        if abs(f_mid) < 1e-10:
            return mid
        if f_low * f_mid <= 0:
            hi = mid
            f_high = f_mid
        else:
            lo = mid
            f_low = f_mid
    return 0.5 * (lo + hi)


def price_leg(
    leg: Leg,
    bucket_surface: pd.DataFrame,
    spot: float,
    rd: float,
    rf: float,
    maturity_years: float,
    method: str,
) -> tuple[float, dict]:
    if leg.option_type == "forward":
        return 0.0, {
            "Leg": leg.name,
            "Side": "Forward",
            "Type": "Forward",
            "Strike": leg.strike,
            "Vol": np.nan,
            "Premium USD/EUR": 0.0,
        }

    vol = surface_vol_pct(bucket_surface, spot, rd, rf, maturity_years, leg.strike, method)
    option_price = fx_option_price(spot, leg.strike, maturity_years, rd, rf, vol, leg.option_type)
    signed_price = leg.quantity * option_price
    return signed_price, {
        "Leg": leg.name,
        "Side": "Long" if leg.quantity > 0 else "Short",
        "Type": leg.option_type.capitalize(),
        "Strike": leg.strike,
        "Vol": vol,
        "Premium USD/EUR": signed_price,
    }


def price_structure(
    legs: list[Leg],
    bucket_surface: pd.DataFrame,
    spot: float,
    rd: float,
    rf: float,
    maturity_years: float,
    method: str,
) -> tuple[float, list[dict]]:
    premium = 0.0
    rows = []
    for leg in legs:
        leg_premium, row = price_leg(leg, bucket_surface, spot, rd, rf, maturity_years, method)
        premium += leg_premium
        rows.append(row)
    return premium, rows


def solve_zero_cost_strike(
    side: str,
    protection_strike: float,
    bucket_surface: pd.DataFrame,
    spot: float,
    rd: float,
    rf: float,
    maturity_years: float,
    method: str,
) -> float | None:
    forward = forward_rate(spot, rd, rf, maturity_years)

    if side == "importer":
        protection_vol = surface_vol_pct(bucket_surface, spot, rd, rf, maturity_years, protection_strike, method)
        target = fx_option_price(spot, protection_strike, maturity_years, rd, rf, protection_vol, "put")

        def objective(call_strike: float) -> float:
            vol = surface_vol_pct(bucket_surface, spot, rd, rf, maturity_years, call_strike, method)
            return fx_option_price(spot, call_strike, maturity_years, rd, rf, vol, "call") - target

        low = max(protection_strike * 1.001, min(spot, forward) * 0.995)
        high = max(spot, forward) * 1.08
        for _ in range(8):
            solved = bisection(objective, low, high)
            if solved is not None:
                return solved
            high *= 1.18
        return None

    protection_vol = surface_vol_pct(bucket_surface, spot, rd, rf, maturity_years, protection_strike, method)
    target = fx_option_price(spot, protection_strike, maturity_years, rd, rf, protection_vol, "call")

    def objective(put_strike: float) -> float:
        vol = surface_vol_pct(bucket_surface, spot, rd, rf, maturity_years, put_strike, method)
        return fx_option_price(spot, put_strike, maturity_years, rd, rf, vol, "put") - target

    low = min(spot, forward) * 0.60
    high = min(protection_strike * 0.999, max(spot, forward) * 1.005)
    return bisection(objective, low, high)


def notional_in_eur(notional: float, notional_ccy: str, spot: float) -> tuple[float, float]:
    if notional_ccy == "EUR":
        return notional, notional * spot
    return notional / spot, notional


def get_bucket_strike(bucket_surface: pd.DataFrame, tenor: str, bucket: str) -> float:
    row = bucket_surface[(bucket_surface["tenor"] == tenor) & (bucket_surface["bucket"] == bucket)]
    if row.empty:
        raise ValueError(f"Missing {bucket} for tenor {tenor}")
    return float(row.iloc[0]["strike"])


def structure_score(name: str, premium_pct: float) -> float:
    base = {
        "Forward": (5.0, 1.0, 5.0),
        "Vanilla": (5.0, 5.0, 5.0),
        "Collar": (4.2, 2.7, 4.6),
        "Risk Reversal": (4.0, 2.6, 4.1),
        "Seagull": (3.4, 2.4, 1.7),
    }
    protection, upside, tail = base[name]
    cost = max(0.0, min(5.0, 5.0 - abs(premium_pct) * 100.0))
    return 0.35 * protection + 0.25 * cost + 0.25 * upside + 0.15 * tail


def make_result(
    name: str,
    legs: list[Leg],
    bucket_surface: pd.DataFrame,
    spot: float,
    rd: float,
    rf: float,
    maturity_years: float,
    method: str,
    notional_eur: float,
    notional_usd_ref: float,
    protection: str,
    upside: str,
    tail_risk: str,
    comment: str,
) -> StructureResult:
    premium_per_eur, leg_rows = price_structure(legs, bucket_surface, spot, rd, rf, maturity_years, method)
    premium_usd = premium_per_eur * notional_eur
    premium_eur = premium_usd / spot
    premium_pct = premium_usd / notional_usd_ref if notional_usd_ref else 0.0
    return StructureResult(
        name=name,
        legs=legs,
        premium_per_eur=premium_per_eur,
        premium_usd=premium_usd,
        premium_eur=premium_eur,
        premium_pct=premium_pct,
        protection=protection,
        upside=upside,
        tail_risk=tail_risk,
        comment=comment,
        score=structure_score(name, premium_pct),
        leg_rows=leg_rows,
    )


def build_structures(
    role: str,
    tenor: str,
    protection_strike: float,
    manual_sold_strike: float,
    tail_strike: float,
    use_zero_cost: bool,
    bucket_surface: pd.DataFrame,
    spot: float,
    rd: float,
    rf: float,
    maturity_years: float,
    method: str,
    notional_eur: float,
    notional_usd_ref: float,
) -> tuple[list[StructureResult], float, str | None]:
    is_importer = role == "USD importer"
    forward = forward_rate(spot, rd, rf, maturity_years)
    solved_warning = None

    if is_importer:
        protection_type = "put"
        sold_type = "call"
        rr_protection_bucket = "25D Put"
        rr_sold_bucket = "25D Call"
        forward_comment = "Bloque le taux d'achat USD, mais supprime tout upside si EUR/USD remonte."
        vanilla_comment = "Protection forte contre une baisse EUR/USD ; la prime dépend de l'ATM vol et du skew downside."
        collar_comment = "Compromis lisible : la protection est financée par la vente d'un upside cap."
        rr_comment = "Structure construite sur les wings 25D ; le coût dépend directement du Risk Reversal."
        seagull_comment = "Réduit encore la prime, mais abandonne une partie de la protection en scénario extrême."
        forward_protection = "Totale"
        vanilla_protection = f"Sous {protection_strike:.4f}"
        upside = "Total"
        collar_upside = "Capé"
        seagull_tail = "Élevé sous le tail strike"
    else:
        protection_type = "call"
        sold_type = "put"
        rr_protection_bucket = "25D Call"
        rr_sold_bucket = "25D Put"
        forward_comment = "Bloque le taux de vente USD, mais supprime l'upside si EUR/USD baisse."
        vanilla_comment = "Protection forte contre une hausse EUR/USD ; la prime dépend de la call wing vol."
        collar_comment = "La protection est financée par la vente d'une participation downside."
        rr_comment = "Structure construite sur les wings 25D ; le coût dépend du Risk Reversal."
        seagull_comment = "Prime plus faible, mais tail risk si EUR/USD dépasse fortement la wing haute."
        forward_protection = "Totale"
        vanilla_protection = f"Au-dessus de {protection_strike:.4f}"
        upside = "Total"
        collar_upside = "Capé"
        seagull_tail = "Élevé au-dessus du tail strike"

    solved_sold_strike = None
    if use_zero_cost:
        solved_sold_strike = solve_zero_cost_strike(
            "importer" if is_importer else "exporter",
            protection_strike,
            bucket_surface,
            spot,
            rd,
            rf,
            maturity_years,
            method,
        )
        if solved_sold_strike is None:
            sold_strike = manual_sold_strike
            solved_warning = "Le solveur zero-cost n'a pas trouvé de strike robuste ; le funding strike manuel est utilisé."
        else:
            sold_strike = solved_sold_strike
    else:
        sold_strike = manual_sold_strike

    rr_protection_strike = get_bucket_strike(bucket_surface, tenor, rr_protection_bucket)
    rr_sold_strike = get_bucket_strike(bucket_surface, tenor, rr_sold_bucket)

    if is_importer:
        seagull_legs = [
            Leg("Protection floor", "put", 1.0, protection_strike),
            Leg("Funding cap", "call", -1.0, sold_strike),
            Leg("Tail give-up", "put", -1.0, tail_strike),
        ]
    else:
        seagull_legs = [
            Leg("Protection cap", "call", 1.0, protection_strike),
            Leg("Funding floor", "put", -1.0, sold_strike),
            Leg("Tail give-up", "call", -1.0, tail_strike),
        ]

    definitions = [
        (
            "Forward",
            [Leg("Forward hedge", "forward", 1.0, forward)],
            forward_protection,
            "Aucun",
            "Faible",
            forward_comment,
        ),
        (
            "Vanilla",
            [Leg("Protection option", protection_type, 1.0, protection_strike)],
            vanilla_protection,
            upside,
            "Faible",
            vanilla_comment,
        ),
        (
            "Collar",
            [
                Leg("Protection option", protection_type, 1.0, protection_strike),
                Leg("Funding option", sold_type, -1.0, sold_strike),
            ],
            vanilla_protection,
            collar_upside,
            "Faible",
            collar_comment,
        ),
        (
            "Risk Reversal",
            [
                Leg(rr_protection_bucket, protection_type, 1.0, rr_protection_strike),
                Leg(rr_sold_bucket, sold_type, -1.0, rr_sold_strike),
            ],
            "Wing 25D",
            collar_upside,
            "Dépend du skew",
            rr_comment,
        ),
        (
            "Seagull",
            seagull_legs,
            "Zone partielle",
            collar_upside,
            seagull_tail,
            seagull_comment,
        ),
    ]

    results = [
        make_result(
            name=name,
            legs=legs,
            bucket_surface=bucket_surface,
            spot=spot,
            rd=rd,
            rf=rf,
            maturity_years=maturity_years,
            method=method,
            notional_eur=notional_eur,
            notional_usd_ref=notional_usd_ref,
            protection=protection,
            upside=upside_text,
            tail_risk=tail_risk,
            comment=comment,
        )
        for name, legs, protection, upside_text, tail_risk, comment in definitions
    ]
    return results, sold_strike, solved_warning


def payoff_per_eur(result: StructureResult, terminal_spots: np.ndarray, role: str) -> np.ndarray:
    payoff = np.zeros_like(terminal_spots, dtype=float)
    is_importer = role == "USD importer"

    for leg in result.legs:
        if leg.option_type == "forward":
            payoff += (leg.strike - terminal_spots) if is_importer else (terminal_spots - leg.strike)
        elif leg.option_type == "call":
            payoff += leg.quantity * np.maximum(terminal_spots - leg.strike, 0.0)
        elif leg.option_type == "put":
            payoff += leg.quantity * np.maximum(leg.strike - terminal_spots, 0.0)

    return payoff - result.premium_per_eur


def structures_table(results: list[StructureResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Structure": result.name,
                "Premium USD": result.premium_usd,
                "Premium EUR": result.premium_eur,
                "Premium % notional": result.premium_pct,
                "Protection": result.protection,
                "Upside": result.upside,
                "Tail risk": result.tail_risk,
                "Score": result.score,
                "Desk comment": result.comment,
            }
            for result in results
        ]
    )


def format_money(value: float, ccy: str) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}{ccy} {abs(value):,.0f}"


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def styled_structure_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Premium USD"] = out["Premium USD"].map(lambda x: format_money(x, "USD"))
    out["Premium EUR"] = out["Premium EUR"].map(lambda x: format_money(x, "EUR"))
    out["Premium % notional"] = out["Premium % notional"].map(format_pct)
    out["Score"] = out["Score"].map(lambda x: f"{x:.2f}/5")
    return out


def role_fr(role: str) -> str:
    return "Importateur USD" if role == "USD importer" else "Exportateur USD"


def display_structure_table(results: list[StructureResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "Produit": result.name,
                "Prime USD": format_money(result.premium_usd, "USD"),
                "Prime EUR": format_money(result.premium_eur, "EUR"),
                "Prime / nominal": format_pct(result.premium_pct),
                "Protection": result.protection,
                "Upside": result.upside,
                "Tail risk": result.tail_risk,
                "Score": f"{result.score:.2f}/5",
                "Lecture desk": result.comment,
            }
        )
    return pd.DataFrame(rows)


def display_market_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    columns = ["tenor", "atm_vol", "rr25", "bf25", "rr10", "bf10", "forward"]
    out = quotes[columns].copy()
    out = out.rename(
        columns={
            "tenor": "Tenor",
            "atm_vol": "ATM vol",
            "rr25": "25D RR",
            "bf25": "25D BF",
            "rr10": "10D RR",
            "bf10": "10D BF",
            "forward": "Forward",
        }
    )
    return out


def display_bucket_matrix(bucket_surface: pd.DataFrame) -> pd.DataFrame:
    pivot = bucket_surface.pivot_table(index="tenor", columns="bucket", values="vol_pct", aggfunc="first")
    tenor_order = (
        bucket_surface[["tenor", "maturity_years"]]
        .drop_duplicates()
        .sort_values("maturity_years")["tenor"]
        .tolist()
    )
    return pivot.loc[tenor_order, BUCKET_LABELS].reset_index().rename(columns={"tenor": "Tenor"})


def display_leg_table(leg_rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(leg_rows).rename(
        columns={
            "Leg": "Leg",
            "Side": "Sens",
            "Type": "Type",
            "Strike": "Strike",
            "Vol": "Vol utilisée",
            "Premium USD/EUR": "Prime USD/EUR",
        }
    )
    out["Sens"] = out["Sens"].replace({"Long": "Long", "Short": "Short", "Forward": "Forward"})
    return out


def make_surface_figure(bucket_surface: pd.DataFrame) -> go.Figure:
    pivot = bucket_surface.pivot_table(index="maturity_years", columns="bucket", values="vol_pct", aggfunc="first")
    pivot = pivot[BUCKET_LABELS]
    tenors = (
        bucket_surface[["maturity_years", "tenor"]]
        .drop_duplicates()
        .sort_values("maturity_years")
        .set_index("maturity_years")
        .loc[pivot.index, "tenor"]
        .tolist()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=np.arange(len(BUCKET_LABELS)),
            y=pivot.index.to_numpy(dtype=float),
            z=pivot.to_numpy(dtype=float),
            colorscale="Turbo",
            colorbar=dict(title="Vol %"),
            hovertemplate="Bucket: %{customdata}<br>Maturité: %{y:.2f}Y<br>Vol: %{z:.3f}%<extra></extra>",
            customdata=np.tile(np.array(BUCKET_LABELS), (len(pivot.index), 1)),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=bucket_surface["bucket_axis"],
            y=bucket_surface["maturity_years"],
            z=bucket_surface["vol_pct"],
            mode="markers",
            marker=dict(size=3, color="#f4f6f8", line=dict(width=0.5, color="#070b0f")),
            text=bucket_surface["tenor"] + " / " + bucket_surface["bucket"],
            hovertemplate="%{text}<br>Strike approx.: %{customdata:.4f}<br>Vol: %{z:.3f}%<extra></extra>",
            customdata=bucket_surface["strike"],
            name="Buckets marché",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=640,
        margin=dict(l=0, r=0, t=28, b=0),
        scene=dict(
            xaxis=dict(title="Delta bucket", tickmode="array", tickvals=list(range(5)), ticktext=BUCKET_LABELS),
            yaxis=dict(title="Maturité", tickmode="array", tickvals=pivot.index.tolist(), ticktext=tenors),
            zaxis=dict(title="Implied vol (%)"),
            camera=dict(eye=dict(x=-1.65, y=-1.7, z=0.9)),
        ),
    )
    return fig


def make_heatmap(bucket_surface: pd.DataFrame) -> go.Figure:
    pivot = bucket_surface.pivot_table(index="tenor", columns="bucket", values="vol_pct", aggfunc="first")
    tenor_order = (
        bucket_surface[["tenor", "maturity_years"]]
        .drop_duplicates()
        .sort_values("maturity_years")["tenor"]
        .tolist()
    )
    pivot = pivot.loc[tenor_order, BUCKET_LABELS]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=BUCKET_LABELS,
            y=tenor_order,
            colorscale="Turbo",
            colorbar=dict(title="Vol %"),
            hovertemplate="%{y} / %{x}<br>Vol: %{z:.3f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Delta bucket",
        yaxis_title="Tenor",
    )
    return fig


def make_payoff_chart(results: list[StructureResult], role: str, spot: float, notional_eur: float) -> go.Figure:
    terminal_spots = np.linspace(spot * 0.86, spot * 1.14, 160)
    fig = go.Figure()
    for result in results:
        payoff_usd = payoff_per_eur(result, terminal_spots, role) * notional_eur
        fig.add_trace(
            go.Scatter(
                x=terminal_spots,
                y=payoff_usd,
                mode="lines",
                name=result.name,
                hovertemplate="Spot final: %{x:.4f}<br>Payoff hedge: USD %{y:,.0f}<extra></extra>",
            )
        )
    fig.add_vline(x=spot, line_width=1, line_dash="dot", line_color="#f6b642")
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=28, b=0),
        xaxis_title="Spot final EUR/USD",
        yaxis_title="Payoff hedge après prime (USD)",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def parse_shocks(text: str) -> list[float]:
    values = []
    for raw in text.replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(float(raw.replace("%", "")) / 100.0)
    return values


def spot_sensitivity_table(
    results: list[StructureResult],
    role: str,
    spot: float,
    notional_eur: float,
    shocks: list[float],
) -> pd.DataFrame:
    rows = []
    for shock in shocks:
        terminal_spot = spot * (1.0 + shock)
        for result in results:
            payoff = payoff_per_eur(result, np.array([terminal_spot]), role)[0] * notional_eur
            rows.append(
                {
                    "Spot shock": shock,
                    "Terminal spot": terminal_spot,
                    "Structure": result.name,
                    "Hedge payoff USD": payoff,
                }
            )
    return pd.DataFrame(rows)


def shock_quotes(quotes: pd.DataFrame, atm_shift: float = 0.0, rr_shift: float = 0.0) -> pd.DataFrame:
    shocked = quotes.copy()
    shocked["atm_vol"] = (shocked["atm_vol"] + atm_shift).clip(lower=0.01)
    shocked["rr25"] = shocked["rr25"] + rr_shift
    shocked["rr10"] = shocked["rr10"] + rr_shift * 2.0
    shocked["vol_25d_call"] = shocked["atm_vol"] + shocked["bf25"] + shocked["rr25"] / 2.0
    shocked["vol_25d_put"] = shocked["atm_vol"] + shocked["bf25"] - shocked["rr25"] / 2.0
    shocked["vol_10d_call"] = shocked["atm_vol"] + shocked["bf10"] + shocked["rr10"] / 2.0
    shocked["vol_10d_put"] = shocked["atm_vol"] + shocked["bf10"] - shocked["rr10"] / 2.0
    return shocked


def premium_sensitivity_table(
    base_results: list[StructureResult],
    quotes: pd.DataFrame,
    spot: float,
    rd: float,
    rf: float,
    maturity_years: float,
    method: str,
    notional_eur: float,
    vol_shift: float,
    skew_shift: float,
) -> pd.DataFrame:
    scenarios = [
        ("Base", 0.0, 0.0),
        (f"ATM vol +{vol_shift:.2f}vp", vol_shift, 0.0),
        (f"ATM vol -{vol_shift:.2f}vp", -vol_shift, 0.0),
        (f"RR {skew_shift:+.2f}vp", 0.0, skew_shift),
        (f"RR {-skew_shift:+.2f}vp", 0.0, -skew_shift),
    ]
    rows = []
    for scenario, atm_shift, rr_shift in scenarios:
        shocked_quotes = shock_quotes(quotes, atm_shift=atm_shift, rr_shift=rr_shift)
        shocked_surface = build_bucket_surface(shocked_quotes, spot)
        for result in base_results:
            premium_per_eur, _ = price_structure(result.legs, shocked_surface, spot, rd, rf, maturity_years, method)
            rows.append(
                {
                    "Scenario": scenario,
                    "Structure": result.name,
                    "Premium USD": premium_per_eur * notional_eur,
                    "Delta vs base USD": premium_per_eur * notional_eur - result.premium_usd,
                }
            )
    return pd.DataFrame(rows)


def make_sensitivity_bar(df: pd.DataFrame) -> go.Figure:
    data = df[df["Scenario"] != "Base"]
    fig = go.Figure()
    for structure, group in data.groupby("Structure"):
        fig.add_trace(
            go.Bar(
                x=group["Scenario"],
                y=group["Delta vs base USD"],
                name=structure,
                hovertemplate="%{x}<br>Variation de prime: USD %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=440,
        margin=dict(l=0, r=0, t=28, b=0),
        yaxis_title="Variation de prime vs base (USD)",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def make_spot_sensitivity_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for structure, group in df.groupby("Structure"):
        fig.add_trace(
            go.Scatter(
                x=group["Terminal spot"],
                y=group["Hedge payoff USD"],
                mode="lines+markers",
                name=structure,
                hovertemplate="Spot: %{x:.4f}<br>Payoff: USD %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(l=0, r=0, t=28, b=0),
        xaxis_title="Spot final EUR/USD",
        yaxis_title="Payoff hedge après prime (USD)",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def coherence_report(quotes: pd.DataFrame, bucket_surface: pd.DataFrame) -> pd.DataFrame:
    wing_cols = ["vol_10d_put", "vol_25d_put", "atm_vol", "vol_25d_call", "vol_10d_call"]
    rows = [
        {
            "Contrôle": "Vols reconstruites positives",
            "Statut": "OK" if (quotes[wing_cols] > 0).all().all() else "À vérifier",
            "Détail": f"Min wing vol = {quotes[wing_cols].min().min():.4f}%",
        },
        {
            "Contrôle": "10D BF supérieur au 25D BF",
            "Statut": "OK" if (quotes["bf10"] >= quotes["bf25"]).all() else "À vérifier",
            "Détail": f"Min BF spread = {(quotes['bf10'] - quotes['bf25']).min():.4f}vp",
        },
        {
            "Contrôle": "|10D RR| supérieur à |25D RR|",
            "Statut": "OK" if (quotes["rr10"].abs() >= quotes["rr25"].abs()).all() else "À vérifier",
            "Détail": f"Min RR spread = {(quotes['rr10'].abs() - quotes['rr25'].abs()).min():.4f}vp",
        },
    ]

    order_ok = True
    for _, group in bucket_surface.groupby("tenor"):
        strikes = group.sort_values("bucket_axis")["strike"].tolist()
        order_ok &= all(strikes[i] < strikes[i + 1] for i in range(len(strikes) - 1))
    rows.append(
        {
            "Contrôle": "Strikes approx. croissants de la put wing à la call wing",
            "Statut": "OK" if order_ok else "À vérifier",
            "Détail": "10D put < 25D put < ATM fwd < 25D call < 10D call",
        }
    )
    return pd.DataFrame(rows)


def desk_read(quotes: pd.DataFrame, tenor: str, role: str) -> str:
    row = quotes[quotes["tenor"] == tenor].iloc[0]
    rr25 = float(row["rr25"])
    bf25 = float(row["bf25"])
    if rr25 < -0.05:
        skew = "Le 25D RR est négatif : les EUR puts sont plus chères que les EUR calls, donc la protection downside EUR est plus difficile à financer."
    elif rr25 > 0.05:
        skew = "Le 25D RR est positif : les EUR calls sont plus chères que les EUR puts, donc la protection upside EUR porte plus de coût de skew."
    else:
        skew = "Le 25D RR est proche de zéro : le funding entre put wing et call wing est plutôt équilibré."

    if bf25 > 0.35:
        smile = "Le Butterfly est élevé : les wings intègrent une prime de convexité et de tail risk."
    elif bf25 < 0.20:
        smile = "Le Butterfly est faible : le smile est assez plat autour des wings 25D."
    else:
        smile = "Le Butterfly est modéré : la prime de smile existe, mais reste contenue."

    role_note = (
        "Pour un importateur USD, la priorité est de se protéger contre une baisse d'EUR/USD."
        if role == "USD importer"
        else "Pour un exportateur USD, la priorité est de se protéger contre une hausse d'EUR/USD."
    )
    return f"{role_note} {skew} {smile}"


def header(spot: float, rd: float, rf: float, quotes: pd.DataFrame, tenor: str) -> None:
    row = quotes[quotes["tenor"] == tenor].iloc[0]
    st.markdown(
        f"""
        <div class="bbg-header">
          <div class="bbg-title">Pricer de produits structurés EUR/USD</div>
          <div class="bbg-subtitle">Surface de volatilité FX, pricing Garman-Kohlhagen et comparaison desk des structures.</div>
          <div class="ticker-row">
            <div class="ticker-cell">PAIR <span class="ticker-value">EUR/USD</span></div>
            <div class="ticker-cell">SPOT <span class="ticker-value">{spot:.4f}</span></div>
            <div class="ticker-cell">USD RATE <span class="ticker-value">{rd:.2%}</span></div>
            <div class="ticker-cell">EUR RATE <span class="ticker-value">{rf:.2%}</span></div>
            <div class="ticker-cell">{tenor} ATM <span class="ticker-value">{row['atm_vol']:.3f}%</span></div>
            <div class="ticker-cell">{tenor} 25D RR <span class="ticker-value">{row['rr25']:.3f}vp</span></div>
            <div class="ticker-cell">{tenor} FWD <span class="ticker-value">{row['forward']:.4f}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Pricer EUR/USD", layout="wide")
    inject_css()

    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Navigation",
        ["Synthèse", "Atelier"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown("## Hypothèses")
    with st.sidebar.expander("Marché", expanded=True):
        uploaded = st.file_uploader("CSV marché", type=["csv"])
        st.caption("Format attendu : tenor, ATM vol, RR/BF 25D et 10D. Le fichier `data.csv` est chargé par défaut.")
        spot = st.number_input("Spot EUR/USD", min_value=0.0001, value=1.0725, step=0.0005, format="%.4f")

    with st.sidebar.expander("Taux et méthode", expanded=False):
        rd = st.number_input("USD rate", value=4.25, step=0.05, format="%.2f", help="Taux domestic pour EUR/USD.") / 100.0
        rf = st.number_input("EUR rate", value=2.75, step=0.05, format="%.2f", help="Taux foreign pour EUR/USD.") / 100.0
        interpolation = st.selectbox("Interpolation surface", ["PCHIP", "Linear"], index=0)
        if interpolation == "PCHIP" and PchipInterpolator is None:
            st.warning("SciPy n'est pas installé. L'interpolation Linear est utilisée.")
            interpolation = "Linear"

    try:
        quotes, source_name = load_quotes(uploaded, spot, rd, rf)
    except Exception as exc:
        st.error(f"Erreur import CSV : {exc}")
        st.stop()

    bucket_surface = build_bucket_surface(quotes, spot)
    tenor_options = quotes["tenor"].tolist()
    default_tenor_index = tenor_options.index("3M") if "3M" in tenor_options else min(6, len(tenor_options) - 1)

    with st.sidebar.expander("Cas client", expanded=True):
        role_label = st.radio("Profil", ["Importateur USD", "Exportateur USD"], horizontal=True)
        role = "USD importer" if role_label == "Importateur USD" else "USD exporter"
        notional_ccy = st.selectbox("Devise du nominal", ["USD", "EUR"], index=0)
        notional = st.number_input("Nominal à couvrir", min_value=1_000.0, value=10_000_000.0, step=100_000.0, format="%.0f")
        tenor = st.selectbox("Maturité de pricing", tenor_options, index=default_tenor_index)
    maturity_years = float(quotes.loc[quotes["tenor"] == tenor, "maturity_years"].iloc[0])

    is_importer = role == "USD importer"
    default_protection = spot * (0.98 if is_importer else 1.02)
    default_sold = spot * (1.03 if is_importer else 0.97)
    default_tail = spot * (0.93 if is_importer else 1.07)
    with st.sidebar.expander("Strikes produits", expanded=False):
        st.caption("Tu peux saisir les strikes à la main. Pour le Collar, le solveur peut calculer automatiquement le funding strike zero-cost.")
        protection_strike = st.number_input(
            "Protection strike",
            min_value=0.0001,
            value=float(default_protection),
            step=0.0005,
            format="%.4f",
        )
        manual_sold_strike = st.number_input(
            "Funding strike manuel",
            min_value=0.0001,
            value=float(default_sold),
            step=0.0005,
            format="%.4f",
        )
        tail_strike = st.number_input("Seagull tail strike", min_value=0.0001, value=float(default_tail), step=0.0005, format="%.4f")
        use_zero_cost = st.checkbox("Calculer le Collar zero-cost", value=True)

    with st.sidebar.expander("Stress tests", expanded=False):
        shock_text = st.text_input("Chocs spot", value="-5%, -2%, 2%, 5%")
        vol_shift = st.number_input("Choc ATM vol, vol points", value=1.00, step=0.25, format="%.2f")
        skew_shift = st.number_input("Choc RR/skew, vol points", value=0.25, step=0.05, format="%.2f")

    notional_eur, notional_usd_ref = notional_in_eur(notional, notional_ccy, spot)
    results, solved_sold_strike, solved_warning = build_structures(
        role=role,
        tenor=tenor,
        protection_strike=protection_strike,
        manual_sold_strike=manual_sold_strike,
        tail_strike=tail_strike,
        use_zero_cost=use_zero_cost,
        bucket_surface=bucket_surface,
        spot=spot,
        rd=rd,
        rf=rf,
        maturity_years=maturity_years,
        method=interpolation,
        notional_eur=notional_eur,
        notional_usd_ref=notional_usd_ref,
    )
    recommendation = max(results, key=lambda item: item.score)

    header(spot, rd, rf, quotes, tenor)

    if solved_warning:
        st.warning(solved_warning)

    st.markdown(f'<div class="desk-note">{desk_read(quotes, tenor, role)}</div>', unsafe_allow_html=True)

    if page == "Synthèse":
        current_row = quotes[quotes["tenor"] == tenor].iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Profil", role_fr(role))
        col2.metric("Nominal réf. USD", format_money(notional_usd_ref, "USD"))
        col3.metric("Maturité", f"{tenor} / {maturity_years:.2f}Y")
        col4.metric("Forward", f"{forward_rate(spot, rd, rf, maturity_years):.4f}")
        col5.metric("Recommandation", recommendation.name)

        col_a, col_b = st.columns([0.92, 1.28])
        with col_a:
            st.markdown("#### Conclusion immédiate")
            st.markdown(
                f"""
                <div class="recommendation">
                <b>{recommendation.name}</b> est la structure proposée pour ce cas.
                Le scoring V1 privilégie une protection forte, une prime contenue, un upside préservé
                et un tail risk limité.
                <br><br>
                Prime estimée : <b>{format_money(recommendation.premium_usd, "USD")}</b>
                ({format_pct(recommendation.premium_pct)} du nominal USD de référence).
                <br><br>
                {recommendation.comment}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### Hypothèses visibles")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Champ": "Spot", "Valeur": f"{spot:.4f}"},
                        {"Champ": f"{tenor} ATM vol", "Valeur": f"{current_row['atm_vol']:.3f}%"},
                        {"Champ": f"{tenor} 25D RR", "Valeur": f"{current_row['rr25']:.3f}vp"},
                        {"Champ": "Forward", "Valeur": f"{forward_rate(spot, rd, rf, maturity_years):.4f}"},
                        {"Champ": "Protection strike", "Valeur": f"{protection_strike:.4f}"},
                        {"Champ": "Funding strike Collar", "Valeur": f"{(solved_sold_strike or manual_sold_strike):.4f}"},
                        {"Champ": "Seagull tail strike", "Valeur": f"{tail_strike:.4f}"},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        with col_b:
            st.markdown("#### Surface de volatilité")
            surface_fig = make_surface_figure(bucket_surface)
            surface_fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(surface_fig, width="stretch")

        st.markdown("#### Comparaison desk")
        st.dataframe(display_structure_table(results), width="stretch", hide_index=True)

        col_c, col_d = st.columns([1.18, 0.82])
        with col_c:
            st.markdown("#### Payoff chart")
            payoff_fig = make_payoff_chart(results, role, spot, notional_eur)
            payoff_fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(payoff_fig, width="stretch")
        with col_d:
            st.markdown("#### Heatmap rapide")
            heatmap_fig = make_heatmap(bucket_surface)
            heatmap_fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(heatmap_fig, width="stretch")

    elif page == "Atelier":
        data_tab, pricer_tab, stress_tab, reglages_tab = st.tabs(["Data", "Pricer", "Stress tests", "Réglages"])

        with data_tab:
            col1, col2, col3, col4 = st.columns(4)
            current_row = quotes[quotes["tenor"] == tenor].iloc[0]
            col1.metric("Fichier", Path(source_name).name)
            col2.metric("Maturités chargées", f"{len(quotes)}")
            col3.metric(f"{tenor} ATM vol", f"{current_row['atm_vol']:.3f}%")
            col4.metric(f"{tenor} 25D RR", f"{current_row['rr25']:.3f}vp")

            st.markdown("#### Quotes de marché")
            st.dataframe(
                display_market_quotes(quotes).style.format(
                    {
                        "ATM vol": "{:.3f}",
                        "25D RR": "{:.3f}",
                        "25D BF": "{:.3f}",
                        "10D RR": "{:.3f}",
                        "10D BF": "{:.3f}",
                        "Forward": "{:.4f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            with st.expander("Contrôles de qualité des données"):
                st.dataframe(coherence_report(quotes, bucket_surface), width="stretch", hide_index=True)

            col_data_a, col_data_b = st.columns([1.05, 0.95])
            with col_data_a:
                st.markdown("#### Surface 3D")
                st.plotly_chart(make_surface_figure(bucket_surface), width="stretch")
            with col_data_b:
                st.markdown("#### Heatmap")
                st.plotly_chart(make_heatmap(bucket_surface), width="stretch")

            with st.expander("Matrice des vols reconstruites"):
                st.dataframe(
                    display_bucket_matrix(bucket_surface).style.format({bucket: "{:.3f}" for bucket in BUCKET_LABELS}),
                    width="stretch",
                    hide_index=True,
                )

        with pricer_tab:
            col1, col2, col3 = st.columns(3)
            col1.metric("Forward rate", f"{forward_rate(spot, rd, rf, maturity_years):.4f}")
            col2.metric("Funding strike Collar", f"{solved_sold_strike:.4f}" if solved_sold_strike else f"{manual_sold_strike:.4f}")
            col3.metric("Maturité pricing", f"{tenor} / {maturity_years:.3f}Y")

            st.markdown(
                f"""
                <div class="recommendation">
                Structure recommandée : <b>{recommendation.name}</b>, score <b>{recommendation.score:.2f}/5</b>.
                La règle privilégie protection forte, prime faible, upside conservé et tail risk limité.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Comparaison des structures")
            st.dataframe(display_structure_table(results), width="stretch", hide_index=True)

            st.markdown("#### Payoff chart")
            st.plotly_chart(make_payoff_chart(results, role, spot, notional_eur), width="stretch")

            with st.expander("Détail des legs"):
                selected_structure = st.selectbox("Produit", [result.name for result in results])
                selected = next(result for result in results if result.name == selected_structure)
                leg_df = display_leg_table(selected.leg_rows)
                st.dataframe(
                    leg_df.style.format({"Strike": "{:.4f}", "Vol utilisée": "{:.3f}", "Prime USD/EUR": "{:.6f}"}),
                    width="stretch",
                    hide_index=True,
                )

        with stress_tab:
            try:
                shocks = parse_shocks(shock_text)
            except ValueError:
                st.error("Les chocs spot doivent être séparés par des virgules, par exemple : -5%, -2%, 2%, 5%.")
                st.stop()

            st.markdown("#### Chocs spot")
            spot_df = spot_sensitivity_table(results, role, spot, notional_eur, shocks)
            st.plotly_chart(make_spot_sensitivity_chart(spot_df), width="stretch")
            display_spot = spot_df.copy()
            display_spot["Spot shock"] = display_spot["Spot shock"].map(format_pct)
            display_spot["Terminal spot"] = display_spot["Terminal spot"].map(lambda x: f"{x:.4f}")
            display_spot["Hedge payoff USD"] = display_spot["Hedge payoff USD"].map(lambda x: format_money(x, "USD"))
            display_spot = display_spot.rename(
                columns={
                    "Spot shock": "Choc spot",
                    "Terminal spot": "Spot final",
                    "Structure": "Produit",
                    "Hedge payoff USD": "Payoff hedge USD",
                }
            )
            st.dataframe(display_spot, width="stretch", hide_index=True)

            st.markdown("#### Chocs ATM vol et RR/skew")
            premium_df = premium_sensitivity_table(
                results,
                quotes,
                spot,
                rd,
                rf,
                maturity_years,
                interpolation,
                notional_eur,
                vol_shift,
                skew_shift,
            )
            st.plotly_chart(make_sensitivity_bar(premium_df), width="stretch")
            display_premium = premium_df.copy()
            display_premium["Premium USD"] = display_premium["Premium USD"].map(lambda x: format_money(x, "USD"))
            display_premium["Delta vs base USD"] = display_premium["Delta vs base USD"].map(lambda x: format_money(x, "USD"))
            display_premium = display_premium.rename(
                columns={
                    "Scenario": "Scénario",
                    "Structure": "Produit",
                    "Premium USD": "Prime USD",
                    "Delta vs base USD": "Variation vs base USD",
                }
            )
            st.dataframe(display_premium, width="stretch", hide_index=True)

        with reglages_tab:
            st.markdown("#### Réglages actifs")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Réglage": "Profil", "Valeur": role_fr(role)},
                        {"Réglage": "Devise nominal", "Valeur": notional_ccy},
                        {"Réglage": "Nominal", "Valeur": f"{notional:,.0f} {notional_ccy}"},
                        {"Réglage": "Spot EUR/USD", "Valeur": f"{spot:.4f}"},
                        {"Réglage": "USD rate", "Valeur": f"{rd:.2%}"},
                        {"Réglage": "EUR rate", "Valeur": f"{rf:.2%}"},
                        {"Réglage": "Interpolation", "Valeur": interpolation},
                        {"Réglage": "Protection strike", "Valeur": f"{protection_strike:.4f}"},
                        {"Réglage": "Funding strike manuel", "Valeur": f"{manual_sold_strike:.4f}"},
                        {"Réglage": "Funding strike utilisé", "Valeur": f"{(solved_sold_strike or manual_sold_strike):.4f}"},
                        {"Réglage": "Seagull tail strike", "Valeur": f"{tail_strike:.4f}"},
                        {"Réglage": "Chocs spot", "Valeur": shock_text},
                        {"Réglage": "Choc ATM vol", "Valeur": f"{vol_shift:.2f}vp"},
                        {"Réglage": "Choc RR/skew", "Valeur": f"{skew_shift:.2f}vp"},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

            st.markdown("#### Extensions possibles")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Module": "Participating Forward",
                            "Statut": "V2",
                            "Utilité desk": "Conserver une partie de l'upside tout en garantissant un hedge rate.",
                        },
                        {
                            "Module": "Simplified TARF",
                            "Statut": "V2",
                            "Utilité desk": "Montrer un produit accrual avec target redemption et risque path-dependent.",
                        },
                        {
                            "Module": "FX delta-to-strike rigoureux",
                            "Statut": "V2",
                            "Utilité desk": "Remplacer l'approximation forward delta par les conventions premium-adjusted.",
                        },
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


if __name__ == "__main__":
    main()
