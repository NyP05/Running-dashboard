import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import csv

st.set_page_config(page_title="Garmin Futás Dashboard", layout="wide")

# =========================================================
# KONFIGURÁCIÓ (magic numberek kiemelve)
# =========================================================
CFG = {
    "speed_bins": 8,
    "tech_weights": {"vr": 0.25, "gct": 0.30, "vo": 0.15, "cad": 0.20, "stride": 0.10},
    "fat_weights": {"gct": 0.35, "vr": 0.30, "cad": 0.20, "hr": 0.15},
    "res_weights": {"hrpw": 0.20, "pps": 0.20, "gct": 0.20, "vr": 0.15, "vo": 0.10, "cad": 0.10, "stride": 0.05},
    "tech_min_group": 12,
    "tech_min_total": 20,
    "fat_min_total": 20,
    "fat_min_baseline": 15,
    "fat_easy_min": 10,
    "fat_easy_full": 30,
    "res_min_total": 25,
    "res_min_group": 12,
    "baseline_weeks_default": 12,
    "baseline_min_runs_default": 25,
    "hrmax_default": 190,
    "ramp_warn": 8.0,
    "ramp_red": 12.0,
    "daily_coach_tech_bad": -5.0,
    "daily_coach_tech_warn": -2.0,
    "daily_coach_fat_bad": 60.0,
    "daily_coach_fat_warn": 45.0,
    "daily_coach_trend_bad": -6.0,
}

# =========================================================
# MOBIL NÉZET
# =========================================================
st.sidebar.divider()
st.sidebar.header("Megjelenés")
MOBILE = st.sidebar.toggle("📱 Mobil nézet", value=True)


def inject_mobile_css(mobile: bool):
    if not mobile:
        return
    st.markdown(
        """
        <style>
        .block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; padding-left: 0.8rem; padding-right: 0.8rem; }
        section[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }
        div[data-testid="stMetric"] { padding: 0.6rem 0.8rem; border-radius: 12px; }
        div[data-testid="stMetric"] label { font-size: 0.85rem !important; }
        div[data-testid="stMetric"] div { font-size: 1.4rem !important; }
        div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
        h1 { font-size: 1.6rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_mobile_css(MOBILE)


def metric_row(mobile: bool, items):
    if mobile:
        cols = st.columns(2)
        for i, it in enumerate(items):
            with cols[i % 2]:
                st.metric(it[0], it[1], delta=it[2] if len(it) > 2 else None)
    else:
        cols = st.columns(len(items))
        for col, it in zip(cols, items):
            col.metric(it[0], it[1], delta=it[2] if len(it) > 2 else None)


# =========================================================
# AUTH
# =========================================================
def require_password():
    if st.session_state.get("auth_ok", False):
        return
    st.title("🔒 Belépés")
    pw = st.text_input("Jelszó", type="password")
    secret = st.secrets.get("APP_PASSWORD", None)
    if secret is None:
        st.error("Hiányzik az APP_PASSWORD secret. (Streamlit Cloud → Settings → Secrets)")
        st.stop()
    if st.button("Belépés"):
        if pw == secret:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Hibás jelszó.")
    st.stop()


require_password()

# =========================================================
# SEGÉDFÜGGVÉNYEK
# =========================================================

def to_float_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"--": np.nan, "": np.nan, "None": np.nan})
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def pace_to_sec_per_km(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace('"', "")
    if x in ("--", ""):
        return np.nan
    try:
        m, s = x.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return np.nan


def pace_str_to_sec(p):
    if pd.isna(p):
        return np.nan
    s = str(p).strip()
    if s in ("", "--", "None", "nan"):
        return np.nan
    try:
        mm, ss = s.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return np.nan


def sec_to_pace_str(sec):
    if pd.isna(sec):
        return "—"
    sec = float(sec)
    mm = int(sec // 60)
    ss = int(round(sec - mm * 60))
    return f"{mm}:{ss:02d}"


def robust_z(x: pd.Series, ref: pd.Series) -> pd.Series:
    refv = ref.to_numpy(dtype=float)
    med = np.nanmedian(refv)
    mad = np.nanmedian(np.abs(refv - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


# safe_col és _col_any egyesítve egyetlen függvénybe
def find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Visszaadja az első létező oszlopnevet a candidates listából."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def classify_from_title(title):
    if not isinstance(title, str):
        return None
    t = title.lower()
    if any(k in t for k in ["easy", "könnyű", "konnyu", "laza"]):
        return "easy"
    if any(k in t for k in ["tempo", "küszöb", "kuszob", "threshold"]):
        return "tempo"
    if any(k in t for k in ["race", "verseny", "maraton", "félmaraton", "felmaraton", "10k", "5k"]):
        return "race"
    return None


def slope_bucket_row(up_m_per_km, down_m_per_km):
    if pd.isna(up_m_per_km) or pd.isna(down_m_per_km):
        return np.nan
    up = float(up_m_per_km)
    down = float(down_m_per_km)
    if up < 5 and down < 5:
        return "flat"
    if max(up, down) < 15:
        return "rolling"
    if up >= 15 and up > down * 1.2:
        return "uphill_dominant"
    if down >= 15 and down > up * 1.2:
        return "downhill_dominant"
    if up >= 15 and down >= 15:
        return "hilly_mixed"
    return "other"


def duration_to_seconds(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in ("", "--", "None", "nan"):
        return np.nan
    s = s.replace(",", ".")
    s_main = s.split(".")[0]
    parts = s_main.split(":")
    try:
        parts = [int(p) for p in parts]
    except Exception:
        return np.nan
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    return np.nan


def hr_zone_from_pct(pct: float) -> str:
    if pd.isna(pct):
        return np.nan
    pct = float(pct)
    if pct < 0.60:
        return "Z1"
    if pct < 0.70:
        return "Z2"
    if pct < 0.80:
        return "Z3"
    if pct < 0.90:
        return "Z4"
    return "Z5"


def run_type_hu(x):
    mapping = {"easy": "Könnyű", "tempo": "Tempó", "race": "Verseny"}
    if pd.isna(x):
        return np.nan
    return mapping.get(str(x).strip().lower(), str(x))


def slope_bucket_hu(x):
    mapping = {
        "flat": "Sík",
        "rolling": "Hullámos",
        "uphill_dominant": "Emelkedő domináns",
        "downhill_dominant": "Lejtő domináns",
        "hilly_mixed": "Dombos vegyes",
        "other": "Egyéb",
    }
    if pd.isna(x):
        return np.nan
    return mapping.get(str(x).strip(), str(x))


def temp_bin(t):
    if pd.isna(t):
        return "unk"
    t = float(t)
    if t < 10:
        return "cold"
    if t < 18:
        return "cool"
    if t < 24:
        return "mild"
    return "hot"


def num(v):
    return pd.to_numeric(
        pd.Series([v]).astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    ).iloc[0]


def med(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    ).median()


def _to_minutes(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in ("--", "", "None", "nan"):
        return np.nan
    try:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 60 + int(m) + int(sec) / 60
        if len(parts) == 2:
            m, sec = parts
            return int(m) + int(sec) / 60
    except Exception:
        return np.nan
    return np.nan


# =========================================================
# BASELINE SEGÉDEK
# =========================================================

def get_type_baseline(
    base_df: pd.DataFrame,
    last_date: pd.Timestamp,
    run_type_col: str | None,
    target_type: str,
    weeks: int = 12,
    min_runs: int = 25,
) -> pd.DataFrame:
    if base_df is None or len(base_df) == 0 or "Dátum" not in base_df.columns:
        return pd.DataFrame()
    dfb = base_df.dropna(subset=["Dátum"]).sort_values("Dátum")
    dfb = dfb[dfb["Dátum"] <= last_date].copy()
    if not run_type_col or run_type_col not in dfb.columns:
        start = last_date - pd.Timedelta(weeks=weeks)
        w = dfb[dfb["Dátum"] >= start].copy()
        return w if len(w) >= min_runs else dfb.tail(min_runs).copy()
    dfb[run_type_col] = dfb[run_type_col].astype(str).str.strip().str.lower()
    t = str(target_type).strip().lower()
    dfb = dfb[dfb[run_type_col] == t].copy()
    if dfb.empty:
        return pd.DataFrame()
    start = last_date - pd.Timedelta(weeks=weeks)
    w = dfb[dfb["Dátum"] >= start].copy()
    return w if len(w) >= min_runs else dfb.tail(min_runs).copy()


def get_easy_baseline(
    base_df: pd.DataFrame,
    last_date: pd.Timestamp,
    weeks: int,
    min_runs: int,
) -> pd.DataFrame:
    easy = (
        base_df[base_df["Run_type"] == "easy"].copy()
        if "Run_type" in base_df.columns
        else base_df.copy()
    )
    easy = easy.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
    if pd.isna(last_date):
        return easy.tail(min_runs).copy()
    start = last_date - pd.Timedelta(weeks=weeks)
    w = easy[(easy["Dátum"] >= start) & (easy["Dátum"] <= last_date)].copy()
    return w if len(w) >= min_runs else easy.tail(min_runs).copy()


# =========================================================
# NAPI COACH ÖSSZEFOGLALÁS
# =========================================================

def daily_coach_summary(
    base_all: pd.DataFrame,
    run_type_col: str | None,
    fatigue_col: str | None,
    baseline_weeks: int,
    baseline_min_runs: int,
) -> tuple[str, str]:
    if base_all is None or len(base_all) == 0:
        return "ℹ️", "Nincs elég adat a napi összképhez."
    b = base_all.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
    if len(b) < 5:
        return "ℹ️", "Nincs elég technika adat (legalább ~5 futás kell)."
    easy = (
        b[b[run_type_col] == "easy"].copy()
        if run_type_col and run_type_col in b.columns
        else b.copy()
    )
    easy = easy.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
    if len(easy) < 5:
        return "ℹ️", "Kevés easy futás → a napi összkép bizonytalan."
    last_easy = easy.iloc[-1]
    baseline_full = get_easy_baseline(b, last_easy["Dátum"], baseline_weeks, baseline_min_runs)
    if len(baseline_full) < 10:
        return "ℹ️", "Kevés easy baseline (min ~10) → még gyűjts adatot 1–2 hétig."
    tech_b = float(np.nanmedian(baseline_full["Technika_index"]))
    tech_last = float(last_easy["Technika_index"])
    tech_delta = tech_last - tech_b
    fat_last = fat_b = None
    if (
        fatigue_col
        and fatigue_col in b.columns
        and baseline_full[fatigue_col].notna().sum() >= 8
    ):
        v = last_easy.get(fatigue_col)
        fat_last = float(v) if pd.notna(v) else None
        fat_b = float(np.nanmedian(baseline_full[fatigue_col]))

    tail7 = easy.tail(7)
    tech_trend = (
        float(tail7["Technika_index"].iloc[-1] - tail7["Technika_index"].iloc[0])
        if len(tail7) >= 4
        else 0.0
    )

    tc_bad = CFG["daily_coach_tech_bad"]
    tc_warn = CFG["daily_coach_tech_warn"]
    fat_bad = CFG["daily_coach_fat_bad"]
    fat_warn_v = CFG["daily_coach_fat_warn"]

    tech_bad = tech_delta < tc_bad
    tech_warn = tc_bad <= tech_delta < tc_warn
    is_fat_bad = fat_last is not None and fat_last >= fat_bad
    is_fat_warn = fat_last is not None and fat_warn_v <= fat_last < fat_bad

    if tech_bad and (is_fat_bad or is_fat_warn):
        status = "🔴"
        msg = f"A technika az easy baseline alatt van (≈{tech_delta:+.1f}), és fáradtabb vagy → holnap inkább pihenő / rövid laza futás + 6×20 mp könnyű repülő."
    elif tech_bad:
        status = "🟠"
        msg = f"A technika az easy baseline alatt van (≈{tech_delta:+.1f}) → holnap legyen könnyebb nap: rövidebb easy, fókusz: ritmus + laza talajfogás."
    elif tech_warn or (is_fat_warn and not tech_bad):
        status = "🟠"
        if fat_last is not None:
            msg = f"Apró ingadozás látszik (tech ≈{tech_delta:+.1f}, fatigue {fat_last:.0f}) → holnap maradj easy-ben és ne emelj terhelést."
        else:
            msg = f"Apró technikai ingadozás (≈{tech_delta:+.1f}) → holnap stabil easy, kontrollált tempóval."
    else:
        status = "🟢"
        if fat_last is not None:
            msg = f"Stabil nap (tech ≈{tech_delta:+.1f} a baseline-hoz, fatigue {fat_last:.0f}) → holnap mehet a terv szerint, de maradj kontrolláltan."
        else:
            msg = f"Stabil nap (tech ≈{tech_delta:+.1f} a baseline-hoz) → holnap mehet a terv, kontrollált easy/tempo aránnyal."

    if tech_trend <= CFG["daily_coach_trend_bad"] and status == "🟢":
        status = "🟠"
        msg = "Az utolsó napokban csúszik a technika trend → holnap inkább könnyebb nap és több regeneráció."
    return status, msg


# =========================================================
# RUNNING ECONOMY (EF / Fatmax / Decoupling)
# =========================================================

def compute_efficiency(
    df: pd.DataFrame,
    hr_col: str | None,
    pace_col: str | None,
    pwr_col: str | None,
) -> pd.DataFrame:
    out = df.copy()
    out["_hr"] = to_float_series(out[hr_col]) if hr_col and hr_col in out.columns else np.nan
    if pace_col and pace_col in out.columns:
        out["_pace_sec"] = out[pace_col].apply(pace_str_to_sec)
        out["_speed_mps"] = np.where(
            out["_pace_sec"].notna() & (out["_pace_sec"] > 0),
            1000.0 / out["_pace_sec"],
            np.nan,
        )
    else:
        out["_speed_mps"] = out["speed_mps"] if "speed_mps" in out.columns else np.nan
    out["_pwr"] = to_float_series(out[pwr_col]) if pwr_col and pwr_col in out.columns else np.nan
    out["EF_pace"] = np.where(
        out["_speed_mps"].notna() & out["_hr"].notna() & (out["_hr"] > 0),
        out["_speed_mps"] / out["_hr"],
        np.nan,
    )
    out["EF_power"] = np.where(
        out["_speed_mps"].notna() & out["_pwr"].notna() & (out["_pwr"] > 0),
        out["_speed_mps"] / out["_pwr"],
        np.nan,
    )
    out["HR_per_speed"] = np.where(
        out["_speed_mps"].notna() & (out["_speed_mps"] > 0),
        out["_hr"] / out["_speed_mps"],
        np.nan,
    )
    out["HR_per_power"] = np.where(
        out["_pwr"].notna() & (out["_pwr"] > 0),
        out["_hr"] / out["_pwr"],
        np.nan,
    )
    return out


def estimate_fatmax_from_runs(
    df_runs: pd.DataFrame,
    hrmax: int,
    hr_col: str,
    pace_col: str,
    pwr_col: str | None = None,
    min_points: int = 12,
):
    if df_runs is None or len(df_runs) == 0:
        return None
    if hr_col not in df_runs.columns or pace_col not in df_runs.columns:
        return None
    x = compute_efficiency(df_runs, hr_col=hr_col, pace_col=pace_col, pwr_col=pwr_col)
    x = x.dropna(subset=["Dátum", "_hr", "_speed_mps"])
    if len(x) < min_points:
        return None
    x["HR_pct"] = x["_hr"] / float(hrmax)
    x = x[(x["HR_pct"] >= 0.55) & (x["HR_pct"] <= 0.80)].copy()
    if len(x) < min_points:
        return None
    bins = np.arange(0.55, 0.80 + 0.0001, 0.02)
    x["HR_bin"] = pd.cut(x["HR_pct"], bins=bins, include_lowest=True)
    use_power = ("EF_power" in x.columns) and (x["EF_power"].notna().sum() >= max(6, min_points // 2))
    ef_col = "EF_power" if use_power else "EF_pace"
    grp = x.groupby("HR_bin")[ef_col].median().dropna()
    if grp.empty:
        return None
    best_bin = grp.idxmax()
    x_best = x[x["HR_bin"] == best_bin].copy()
    if x_best.empty:
        return None
    hr_fatmax = float(np.nanmedian(x_best["_hr"]))
    speed_fatmax = float(np.nanmedian(x_best["_speed_mps"]))
    pace_sec = 1000.0 / speed_fatmax if speed_fatmax > 0 else np.nan
    pace_fatmax = sec_to_pace_str(pace_sec)
    pwr_fatmax = (
        float(np.nanmedian(x_best["_pwr"]))
        if (use_power and x_best["_pwr"].notna().any())
        else np.nan
    )
    lo = float(best_bin.left) if hasattr(best_bin, "left") else np.nan
    hi = float(best_bin.right) if hasattr(best_bin, "right") else np.nan
    hrpct_mid = (lo + hi) / 2.0 if (pd.notna(lo) and pd.notna(hi)) else np.nan
    return {
        "ef_col": ef_col,
        "best_bin": str(best_bin),
        "hrpct_mid": hrpct_mid,
        "hr_fatmax": hr_fatmax,
        "pace_fatmax": pace_fatmax,
        "pwr_fatmax": pwr_fatmax,
        "n_points": int(len(x_best)),
        "table": x,
        "table_best": x_best,
    }


def aerobic_decoupling_proxy(
    last_row: pd.Series,
    baseline_df: pd.DataFrame,
    hr_col: str,
    pace_col: str,
    pwr_col: str | None,
):
    if baseline_df is None or len(baseline_df) < 8:
        return None
    tmp = compute_efficiency(baseline_df, hr_col=hr_col, pace_col=pace_col, pwr_col=pwr_col)
    last_df = compute_efficiency(
        pd.DataFrame([last_row]), hr_col=hr_col, pace_col=pace_col, pwr_col=pwr_col
    )
    use_power = (
        "EF_power" in tmp.columns
        and tmp["EF_power"].notna().sum() >= 8
        and last_df["EF_power"].notna().sum() >= 1
    )
    ef = "EF_power" if use_power else "EF_pace"
    b = float(np.nanmedian(tmp[ef]))
    v = float(last_df[ef].iloc[0]) if pd.notna(last_df[ef].iloc[0]) else np.nan
    if pd.isna(b) or b == 0 or pd.isna(v):
        return None
    return {"ef": ef, "baseline_med": b, "last": v, "delta_pct": float((v - b) / b * 100.0)}


# =========================================================
# ADATBEOLVASÁS (cached)
# =========================================================

@st.cache_data(show_spinner=False)
def load_any(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    import io
    name = file_name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    text = None
    for enc in ["utf-8-sig", "utf-8", "cp1250", "latin1"]:
        try:
            text = file_bytes.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = file_bytes.decode("utf-8", errors="replace")
    lines = text.strip().splitlines()
    if not lines:
        return pd.DataFrame()
    header = lines[0].split(",")
    data = []
    for raw_data in lines[1:]:
        raw_data = raw_data.replace('""', '"')
        if raw_data.startswith('"'):
            raw_data = raw_data[1:]
        reader = csv.reader([raw_data], delimiter=",", quotechar='"')
        row = next(reader)
        data.append(row)
    df = pd.DataFrame(data, columns=header)
    df.columns = pd.Index(df.columns).astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def fix_mojibake_columns(df: pd.DataFrame) -> pd.DataFrame:
    if any("Ă" in c for c in df.columns.astype(str)):
        new_cols = []
        for c in df.columns.astype(str):
            try:
                new_cols.append(c.encode("latin1").decode("utf-8"))
            except Exception:
                new_cols.append(c)
        df = df.copy()
        df.columns = new_cols
    return df


# =========================================================
# TELJES PIPELINE (cached – ez a legnagyobb teljesítmény-nyereség)
# =========================================================

NUM_MAP = {
    "Átl. pedálütem": "cad_num",
    "Átlagos lépéshossz": "stride_num",
    "Átlagos függőleges arány": "vr_num",
    "Átlagos függőleges oszcilláció": "vo_num",
    "Átlagos talajérintési idő": "gct_num",
    "Átlagos pulzusszám": "hr_num",
    "Teljes emelkedés": "asc_m",
    "Teljes süllyedés": "des_m",
    "Távolság": "dist_km",
}


@st.cache_data(show_spinner="Adatok feldolgozása…")
def full_pipeline(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """
    Teljes adatfeldolgozás egyszerre, cache-elve.
    Csak akkor fut újra, ha a fájl megváltozik.
    """
    df0 = load_any(file_bytes, file_name)
    df0 = fix_mojibake_columns(df0)
    df = df0.copy()

    # --- Dátum parse
    date_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["dátum", "datum", "date"])
    ]
    if date_candidates:
        s = (
            df[date_candidates[0]]
            .astype(str).str.strip()
            .replace({"--": np.nan, "": np.nan, "None": np.nan})
        )
        dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S")
        if dt.notna().sum() == 0:
            dt = pd.to_datetime(s, errors="coerce")
        df["Dátum"] = dt
    else:
        df["Dátum"] = pd.NaT

    # --- Futás szűrő maszk
    mask_run = (
        df["Tevékenység típusa"].astype(str).str.contains("Fut", na=False)
        if "Tevékenység típusa" in df.columns
        else pd.Series(True, index=df.index)
    )

    # --- Numerikus oszlopok
    for src, dst in NUM_MAP.items():
        df[dst] = to_float_series(df[src]) if src in df.columns else np.nan

    df["pace_sec_km"] = (
        df["Átlagos tempó"].apply(pace_to_sec_per_km)
        if "Átlagos tempó" in df.columns
        else np.nan
    )
    df["speed_mps"] = np.where(
        df["pace_sec_km"].notna() & (df["pace_sec_km"] > 0),
        1000.0 / df["pace_sec_km"],
        np.nan,
    )

    # --- Slope feature-ök
    df["up_m_per_km"] = np.where(
        df["dist_km"].notna() & (df["dist_km"] > 0), df["asc_m"] / df["dist_km"], np.nan
    )
    df["down_m_per_km"] = np.where(
        df["dist_km"].notna() & (df["dist_km"] > 0), df["des_m"] / df["dist_km"], np.nan
    )
    df["net_elev_m"] = df["asc_m"] - df["des_m"]
    df["slope_bucket"] = [
        slope_bucket_row(u, d2) for u, d2 in zip(df["up_m_per_km"], df["down_m_per_km"])
    ]

    # --- Run_type (cím + pace fallback)
    df["Run_type"] = df["Cím"].apply(classify_from_title) if "Cím" in df.columns else None

    valid_pace = df.loc[mask_run & df["pace_sec_km"].notna(), "pace_sec_km"]
    if valid_pace.notna().sum() >= 30:
        p40 = np.nanpercentile(valid_pace, 40)
        p80 = np.nanpercentile(valid_pace, 80)

        def classify_from_pace(row):
            if pd.notna(row["Run_type"]):
                return row["Run_type"]
            ps = row["pace_sec_km"]
            if pd.isna(ps):
                return None
            if ps >= p80:
                return "easy"
            if ps >= p40:
                return "tempo"
            return "race"

        df["Run_type"] = df.apply(classify_from_pace, axis=1)

    if "Run_type" in df.columns:
        df["Run_type"] = (
            df["Run_type"].astype(str).str.strip().str.lower()
            .replace({"none": np.nan, "nan": np.nan, "": np.nan})
        )

    # --- Interval típus bevezetése (Power_fatigue_hint alapján)
    df["power_avg_w"] = (
        to_float_series(df["Átl. teljesítmény"]) if "Átl. teljesítmény" in df.columns else np.nan
    )
    df["power_max_w"] = (
        to_float_series(df["Max. teljesítmény"]) if "Max. teljesítmény" in df.columns else np.nan
    )
    df["Power_fatigue_hint"] = np.where(
        df["power_avg_w"].notna() & df["power_max_w"].notna() & (df["power_avg_w"] > 0),
        df["power_max_w"] / df["power_avg_w"],
        np.nan,
    )
    # Ha max/avg > 2.0 és Run_type nem race → interval
    interval_mask = (
        df["Power_fatigue_hint"].notna()
        & (df["Power_fatigue_hint"] >= 2.0)
        & df["Run_type"].isin(["easy", "tempo", np.nan, None])
    )
    df.loc[interval_mask, "Run_type"] = "interval"

    # --- Temperature
    temp_col = find_col(df, "Hőmérséklet", "Temperature", "Temp", "Átlag hőmérséklet", "Avg Temperature")
    df["temp_c"] = to_float_series(df[temp_col]) if temp_col else np.nan
    df["temp_bin"] = df["temp_c"].apply(temp_bin)
    df["hr_avg"] = df["hr_num"] if "hr_num" in df.columns else np.nan

    # --- hr_per_watt, power_per_speed
    df["hr_per_watt"] = np.where(
        df["hr_avg"].notna() & df["power_avg_w"].notna() & (df["power_avg_w"] > 0),
        df["hr_avg"] / df["power_avg_w"],
        np.nan,
    )
    df["power_per_speed"] = np.where(
        df["power_avg_w"].notna() & df["speed_mps"].notna() & (df["speed_mps"] > 0),
        df["power_avg_w"] / df["speed_mps"],
        np.nan,
    )

    # =========================================================
    # TECHNIKA_INDEX (slope-aware + speed bin) – vektorizált
    # =========================================================
    df["Technika_index"] = np.nan
    tech_base = df[mask_run & df["speed_mps"].notna()].copy()

    if len(tech_base) >= CFG["tech_min_total"]:
        tech_base["speed_bin"] = pd.qcut(
            tech_base["speed_mps"], q=CFG["speed_bins"], duplicates="drop"
        )
        for col in ["skill_vr", "skill_gct", "skill_vo", "skill_cad", "skill_stride"]:
            tech_base[col] = np.nan

        def _fill_tech_group(g: pd.DataFrame, target: pd.DataFrame):
            idx = g.index
            min_n = CFG["tech_min_group"]
            if g["vr_num"].notna().sum() >= min_n:
                target.loc[idx, "skill_vr"] = -robust_z(g["vr_num"], g["vr_num"])
            if g["gct_num"].notna().sum() >= min_n:
                target.loc[idx, "skill_gct"] = -robust_z(g["gct_num"], g["gct_num"])
            if g["vo_num"].notna().sum() >= min_n:
                target.loc[idx, "skill_vo"] = -robust_z(g["vo_num"], g["vo_num"])
            if g["cad_num"].notna().sum() >= min_n:
                target.loc[idx, "skill_cad"] = -np.abs(robust_z(g["cad_num"], g["cad_num"]))
            if g["stride_num"].notna().sum() >= min_n:
                target.loc[idx, "skill_stride"] = robust_z(g["stride_num"], g["stride_num"])

        grouped = tech_base.groupby(["speed_bin", "slope_bucket"], dropna=False)
        for _, g in grouped:
            if len(g) >= CFG["tech_min_group"]:
                _fill_tech_group(g, tech_base)

        still_nan = (
            tech_base["skill_vr"].isna()
            & tech_base["skill_gct"].isna()
            & tech_base["skill_vo"].isna()
            & tech_base["skill_cad"].isna()
        )
        if still_nan.any():
            for _, g in tech_base[still_nan].groupby("speed_bin"):
                _fill_tech_group(g, tech_base)

        w = CFG["tech_weights"]
        raw = (
            w["vr"] * tech_base["skill_vr"].fillna(0)
            + w["gct"] * tech_base["skill_gct"].fillna(0)
            + w["vo"] * tech_base["skill_vo"].fillna(0)
            + w["cad"] * tech_base["skill_cad"].fillna(0)
            + w["stride"] * tech_base["skill_stride"].fillna(0)
        )
        p5, p95 = np.nanpercentile(raw, 5), np.nanpercentile(raw, 95)
        tech_base["Technika_index"] = (100 * (raw - p5) / (p95 - p5 + 1e-9)).clip(0, 100)
        df.loc[tech_base.index, "Technika_index"] = tech_base["Technika_index"].values

    # =========================================================
    # FATIGUE_SCORE – vektorizált (iterrows helyett groupby-apply)
    # =========================================================
    df["Fatigue_score"] = np.nan
    df["Fatigue_flag"] = np.nan
    df["Fatigue_type"] = np.nan

    fat = df[mask_run & df["Technika_index"].notna()].copy()
    fat["hr_per_pace"] = np.where(
        fat["hr_num"].notna() & fat["pace_sec_km"].notna() & (fat["pace_sec_km"] > 0),
        fat["hr_num"] / fat["pace_sec_km"],
        np.nan,
    )

    if len(fat) >= CFG["fat_min_total"]:
        easy_base = fat[fat["Run_type"] == "easy"].copy() if "Run_type" in fat.columns else fat.copy()

        def _compute_fat_z_vectorized(
            fat_df: pd.DataFrame, easy_df: pd.DataFrame
        ) -> pd.DataFrame:
            """
            Vektorizált fatigue z-score számítás.
            Minden slope_bucket-hez egy baseline; ha kevés, fallback az egész easy_base.
            """
            result = fat_df.copy()
            for col in ["fatigue_gct", "fatigue_vr", "fatigue_cad", "fatigue_hr"]:
                result[col] = 0.0

            def _get_base(slope_val):
                if len(easy_df) >= CFG["fat_easy_full"] and pd.notna(slope_val):
                    sub = easy_df[easy_df["slope_bucket"] == slope_val]
                    if len(sub) >= 20:
                        return sub
                if len(easy_df) >= CFG["fat_easy_min"]:
                    return easy_df
                return fat_df

            for sb, group in fat_df.groupby("slope_bucket", dropna=False):
                base = _get_base(sb)
                idx = group.index

                for fat_col, src_col, invert in [
                    ("fatigue_gct", "gct_num", False),
                    ("fatigue_vr", "vr_num", False),
                    ("fatigue_hr", "hr_per_pace", False),
                ]:
                    if (
                        group[src_col].notna().sum() >= 5
                        and base[src_col].notna().sum() >= CFG["fat_min_baseline"]
                    ):
                        z = robust_z(group[src_col], base[src_col])
                        result.loc[idx, fat_col] = z.fillna(0).values

                if (
                    group["cad_num"].notna().sum() >= 5
                    and base["cad_num"].notna().sum() >= CFG["fat_min_baseline"]
                ):
                    z = robust_z(group["cad_num"], base["cad_num"])
                    result.loc[idx, "fatigue_cad"] = np.abs(z.fillna(0).values)

            return result

        fat = _compute_fat_z_vectorized(fat, easy_base)

        fw = CFG["fat_weights"]
        raw_f = (
            fw["gct"] * fat["fatigue_gct"]
            + fw["vr"] * fat["fatigue_vr"]
            + fw["cad"] * fat["fatigue_cad"]
            + fw["hr"] * fat["fatigue_hr"]
        )
        p5, p95 = np.nanpercentile(raw_f, 5), np.nanpercentile(raw_f, 95)
        fat["Fatigue_score"] = (100 * (raw_f - p5) / (p95 - p5 + 1e-9)).clip(0, 100)
        fat["Fatigue_flag"] = fat["Fatigue_score"] > 65

        mech = fat["fatigue_gct"] + fat["fatigue_vr"]
        cardio = fat["fatigue_hr"]
        conditions = [
            (mech > 1.5) & (cardio < 0.5),
            (cardio > 1.2) & (mech < 1.0),
            (mech > 1.2) & (cardio > 1.0),
        ]
        choices = ["mechanical", "cardio", "mixed"]
        fat["Fatigue_type"] = np.select(conditions, choices, default="none")

        df.loc[fat.index, "Fatigue_score"] = fat["Fatigue_score"].values
        df.loc[fat.index, "Fatigue_flag"] = fat["Fatigue_flag"].values
        df.loc[fat.index, "Fatigue_type"] = fat["Fatigue_type"].values

    # =========================================================
    # RES+ (Running Economy Score)
    # =========================================================
    df["RES_plus"] = np.nan
    eco = df[mask_run].copy()
    eco = eco.dropna(subset=["Dátum", "speed_mps"])

    if len(eco) >= CFG["res_min_total"]:
        eco["speed_bin"] = pd.qcut(eco["speed_mps"], q=CFG["speed_bins"], duplicates="drop")
        group_cols = ["speed_bin", "slope_bucket", "temp_bin"]
        for c in ["eco_hrpw", "eco_pps", "eco_gct", "eco_vr", "eco_vo", "eco_cad", "eco_stride"]:
            eco[c] = np.nan

        def _fill_eco_group(g: pd.DataFrame, target: pd.DataFrame):
            idx = g.index
            min_n = CFG["res_min_group"]
            if g["hr_per_watt"].notna().sum() >= min_n:
                target.loc[idx, "eco_hrpw"] = -robust_z(g["hr_per_watt"], g["hr_per_watt"])
            if g["power_per_speed"].notna().sum() >= min_n:
                target.loc[idx, "eco_pps"] = -robust_z(g["power_per_speed"], g["power_per_speed"])
            if g["gct_num"].notna().sum() >= min_n:
                target.loc[idx, "eco_gct"] = -robust_z(g["gct_num"], g["gct_num"])
            if g["vr_num"].notna().sum() >= min_n:
                target.loc[idx, "eco_vr"] = -robust_z(g["vr_num"], g["vr_num"])
            if g["vo_num"].notna().sum() >= min_n:
                target.loc[idx, "eco_vo"] = -robust_z(g["vo_num"], g["vo_num"])
            if g["cad_num"].notna().sum() >= min_n:
                target.loc[idx, "eco_cad"] = -np.abs(robust_z(g["cad_num"], g["cad_num"]))
            if g["stride_num"].notna().sum() >= min_n:
                target.loc[idx, "eco_stride"] = robust_z(g["stride_num"], g["stride_num"])

        for _, g in eco.groupby(group_cols, dropna=False):
            if len(g) >= CFG["res_min_group"]:
                _fill_eco_group(g, eco)

        still_nan = (
            eco["eco_gct"].isna() & eco["eco_vr"].isna()
            & eco["eco_vo"].isna() & eco["eco_cad"].isna()
        )
        if still_nan.any():
            for _, g in eco[still_nan].groupby(["speed_bin", "slope_bucket"], dropna=False):
                if len(g) >= CFG["res_min_group"]:
                    _fill_eco_group(g, eco)

        rw = CFG["res_weights"]
        raw = (
            rw["hrpw"] * eco["eco_hrpw"].fillna(0)
            + rw["pps"] * eco["eco_pps"].fillna(0)
            + rw["gct"] * eco["eco_gct"].fillna(0)
            + rw["vr"] * eco["eco_vr"].fillna(0)
            + rw["vo"] * eco["eco_vo"].fillna(0)
            + rw["cad"] * eco["eco_cad"].fillna(0)
            + rw["stride"] * eco["eco_stride"].fillna(0)
        )
        p5, p95 = np.nanpercentile(raw, 5), np.nanpercentile(raw, 95)
        eco["RES_plus"] = (100 * (raw - p5) / (p95 - p5 + 1e-9)).clip(0, 100)
        df.loc[eco.index, "RES_plus"] = eco["RES_plus"].values

    return df


# =========================================================
# ADATFORRÁS + PIPELINE HÍVÁS
# =========================================================
st.sidebar.header("Adatforrás")
uploaded = st.sidebar.file_uploader(
    "Tölts fel Garmin exportot (XLSX ajánlott)", type=["xlsx", "csv"]
)

if uploaded is None:
    st.title("🏃 Garmin Futás Dashboard")
    st.info("Tölts fel egy XLSX vagy CSV Garmin exportot. (Nem mentjük el az adatokat.)")
    st.stop()

# Cache-elt pipeline: csak fájl-változáskor fut újra
file_bytes = uploaded.getvalue()
df = full_pipeline(file_bytes, uploaded.name)

# =========================================================
# ALAP SZŰRÉS + MEGJELENÍTÉSI ELŐKÉSZÍTÉS
# =========================================================
st.title("🏃 Garmin Futás Dashboard")

if "Dátum" not in df.columns:
    st.error("Nem találom a 'Dátum' oszlopot.")
    st.stop()

d = df[df["Dátum"].notna()].copy()
if d.empty:
    st.error("Nincs érvényes dátummal rendelkező sor (Dátum parse -> NaT).")
    st.stop()

run_type_col = find_col(d, "Run_type")
fatigue_col = find_col(d, "Fatigue_score")
fatigue_type_col = find_col(d, "Fatigue_type")
slope_col = find_col(d, "slope_bucket")

if run_type_col:
    d["Edzés típusa"] = d[run_type_col].apply(run_type_hu)
if slope_col:
    d["Terep"] = d[slope_col].apply(slope_bucket_hu)

# =========================================================
# SIDEBAR: HRmax + Szűrők + Baseline
# =========================================================
st.sidebar.divider()
st.sidebar.header("Intenzitás (HR zónák)")
if "hrmax" not in st.session_state:
    st.session_state.hrmax = CFG["hrmax_default"]

st.session_state.hrmax = st.sidebar.number_input(
    "HRmax (ütés/perc) – zónákhoz",
    min_value=120,
    max_value=240,
    value=int(st.session_state.hrmax),
    step=1,
    help="Ha nem tudod pontosan, hagyd 185–195 körül.",
)
hrmax = int(st.session_state.hrmax)

st.sidebar.header("Szűrők")
min_date = pd.to_datetime(d["Dátum"], errors="coerce").dropna().min()
max_date = pd.to_datetime(d["Dátum"], errors="coerce").dropna().max()

if pd.isna(min_date) or pd.isna(max_date):
    st.sidebar.error("Nem sikerült dátumtartományt képezni (NaT).")
    st.stop()

dt_input = st.sidebar.date_input("Dátumtartomány", value=(min_date.date(), max_date.date()))
date_from, date_to = (dt_input[0], dt_input[1]) if len(dt_input) == 2 else (dt_input[0], dt_input[0])

mask = (d["Dátum"].dt.date >= date_from) & (d["Dátum"].dt.date <= date_to)

if run_type_col:
    types = ["easy", "tempo", "race", "interval"]
    present = [t for t in types if t in set(d[run_type_col].dropna().astype(str))]
    selected_types = st.sidebar.multiselect(
        "Edzés típusa",
        options=present,
        default=present,
        format_func=lambda x: {"easy": "Könnyű", "tempo": "Tempó", "race": "Verseny", "interval": "Interval"}.get(x, x),
    )
    if selected_types:
        mask &= d[run_type_col].isin(selected_types)

if slope_col:
    slope_opts = d[slope_col].dropna().unique().tolist()
    slope_sel = st.sidebar.multiselect(
        "Terep", options=slope_opts, default=slope_opts, format_func=slope_bucket_hu
    )
    if slope_sel:
        mask &= d[slope_col].isin(slope_sel)

if fatigue_col and d[fatigue_col].notna().sum() > 0:
    fmin = float(np.nanmin(d[fatigue_col]))
    fmax = float(np.nanmax(d[fatigue_col]))
    sel_fmin, sel_fmax = st.sidebar.slider("Fatigue_score", min_value=fmin, max_value=fmax, value=(fmin, fmax))
    mask &= d[fatigue_col].between(sel_fmin, sel_fmax)

st.sidebar.divider()
st.sidebar.header("Baseline (B-line)")
baseline_mode = st.sidebar.selectbox(
    "Baseline mód",
    options=["Auto (Edzés típusa szerint)", "Mindig EASY baseline"],
    index=0,
)
baseline_weeks = st.sidebar.slider("Baseline ablak (hetek)", min_value=4, max_value=52, value=CFG["baseline_weeks_default"], step=1)
baseline_min_runs = st.sidebar.slider("Minimum baseline futás", min_value=10, max_value=80, value=CFG["baseline_min_runs_default"], step=5)

# Export gomb
st.sidebar.divider()
st.sidebar.header("Export")
view_for_export = d.loc[mask].copy().sort_values("Dátum")
csv_export = view_for_export.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="⬇️ Szűrt adatok letöltése (CSV)",
    data=csv_export,
    file_name="garmin_filtered.csv",
    mime="text/csv",
)

# Kijelentkezés
st.sidebar.divider()
if st.sidebar.button("Kijelentkezés"):
    st.session_state.auth_ok = False
    st.rerun()

view = d.loc[mask].copy().sort_values("Dátum")

# =========================================================
# TABOK
# =========================================================
tab_overview, tab_last, tab_warn, tab_ready, tab_data = st.tabs(
    ["📌 Áttekintés", "🔎 Utolsó futás", "🚦 Warning", "🏁 Readiness", "📄 Adatok"]
)

# =========================================================
# TAB: ÁTTEKINTÉS
# =========================================================
with tab_overview:
    # --- Napi coach összefoglalás
    st.subheader("🗓️ Napi összkép (coach)")
    status, msg = daily_coach_summary(
        base_all=d,
        run_type_col=run_type_col,
        fatigue_col=fatigue_col,
        baseline_weeks=baseline_weeks,
        baseline_min_runs=baseline_min_runs,
    )
    if status == "🔴":
        st.error(f"{status} {msg}")
    elif status == "🟠":
        st.warning(f"{status} {msg}")
    else:
        st.success(f"{status} {msg}")

    st.divider()

    # --- KPI sor
    tech_avg = view["Technika_index"].mean() if ("Technika_index" in view.columns and view["Technika_index"].notna().any()) else np.nan
    fat_avg = view[fatigue_col].mean() if (fatigue_col and view[fatigue_col].notna().any()) else np.nan
    most_type = run_type_hu(view[run_type_col].value_counts().index[0]) if (run_type_col and len(view) > 0) else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Futások (szűrve)", f"{len(view)}")
    c2.metric("Átlag Technika_index", f"{tech_avg:.1f}" if pd.notna(tech_avg) else "—")
    c3.metric("Átlag Fatigue_score", f"{fat_avg:.1f}" if pd.notna(fat_avg) else "—")
    c4.metric("Leggyakoribb edzés típus", most_type)

    st.divider()

    # --- Heti terhelés & ramp rate
    st.subheader("📊 Heti terhelés & ramp rate")
    base_all = d[d["Dátum"].notna()].sort_values("Dátum").copy()

    time_col_candidates = [c for c in ["Idő", "Menetidő", "Eltelt idő"] if c in base_all.columns]
    if time_col_candidates:
        base_all["dur_sec"] = base_all[time_col_candidates[0]].apply(duration_to_seconds)
    else:
        base_all["dur_sec"] = np.nan

    weekly = (
        base_all.set_index("Dátum")
        .resample("W-MON")
        .agg(week_km=("dist_km", "sum"), week_sec=("dur_sec", "sum"), week_elev=("asc_m", "sum"))
        .reset_index()
        .rename(columns={"Dátum": "week"})
        .sort_values("week")
    )
    weekly["week_hours"] = weekly["week_sec"] / 3600.0

    rr_km = rr_time = rr_elev = np.nan
    if len(weekly) >= 6:
        last_idx = len(weekly) - 1
        if weekly.loc[last_idx, "week_km"] < 0.3 * max(1e-9, weekly.loc[last_idx - 1, "week_km"]):
            last_idx -= 1
        last_week = weekly.loc[last_idx]
        prev4 = weekly.loc[last_idx - 4 : last_idx - 1]

        def ramp(curr, prev_mean):
            if pd.isna(curr) or pd.isna(prev_mean) or prev_mean <= 0:
                return np.nan
            return (curr - prev_mean) / prev_mean * 100.0

        rr_km = ramp(float(last_week["week_km"]), float(np.nanmean(prev4["week_km"])))
        rr_time = ramp(float(last_week["week_hours"]), float(np.nanmean(prev4["week_hours"])))
        rr_elev = ramp(float(last_week["week_elev"]), float(np.nanmean(prev4["week_elev"])))

    metric = st.selectbox("Melyik terhelést nézzük?", ["Heti km", "Heti idő (óra)", "Heti emelkedés (m)"], key="load_metric")
    y_col = {"Heti km": "week_km", "Heti idő (óra)": "week_hours", "Heti emelkedés (m)": "week_elev"}[metric]
    rr_val = {"Heti km": rr_km, "Heti idő (óra)": rr_time, "Heti emelkedés (m)": rr_elev}[metric]
    ylabel = {"Heti km": "km", "Heti idő (óra)": "óra", "Heti emelkedés (m)": "m"}[metric]

    def ramp_badge(rr):
        if pd.isna(rr):
            return ("⚪", "Ramp rate: nincs elég adat (legalább 6 hét kell)")
        if rr <= CFG["ramp_warn"]:
            return ("🟢", f"Ramp rate: {rr:+.1f}% (biztonságos)")
        if rr <= CFG["ramp_red"]:
            return ("🟠", f"Ramp rate: {rr:+.1f}% (figyelmeztető)")
        return ("🔴", f"Ramp rate: {rr:+.1f}% (túl gyors emelés)")

    badge, badge_txt = ramp_badge(rr_val)
    st.caption(f"{badge} {badge_txt}")
    st.plotly_chart(
        px.bar(weekly, x="week", y=y_col, title=f"Heti terhelés – {metric}", labels={"week": "Hét", y_col: ylabel}),
        use_container_width=True,
    )

    # --- Gördülő trend
    daily = base_all.copy()
    daily["date"] = daily["Dátum"].dt.date
    daily = daily.groupby("date", as_index=False).agg(km=("dist_km", "sum"), sec=("dur_sec", "sum"), elev=("asc_m", "sum"))
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["km_7"] = daily["km"].rolling(7, min_periods=3).sum()
    daily["km_28"] = daily["km"].rolling(28, min_periods=10).sum()
    daily["h_7"] = daily["sec"].rolling(7, min_periods=3).sum() / 3600.0
    daily["h_28"] = daily["sec"].rolling(28, min_periods=10).sum() / 3600.0
    daily["elev_7"] = daily["elev"].rolling(7, min_periods=3).sum()
    daily["elev_28"] = daily["elev"].rolling(28, min_periods=10).sum()

    roll_cols = {
        "Heti km": ["km_7", "km_28"],
        "Heti idő (óra)": ["h_7", "h_28"],
        "Heti emelkedés (m)": ["elev_7", "elev_28"],
    }[metric]
    st.plotly_chart(
        px.line(daily, x="date", y=roll_cols, title="Gördülő összeg – 7 nap vs 28 nap"),
        use_container_width=True,
    )

    with st.expander("📋 Heti táblázat (részletek)"):
        st.dataframe(weekly.tail(24), use_container_width=True, hide_index=True)

    # --- Éves aktivitás
    st.divider()
    st.subheader("📅 Éves aktivitás (naptár)")
    if "Dátum" in d.columns and "dist_km" in d.columns:
        cal_df = d.dropna(subset=["Dátum"]).copy()
        cal_df["year"] = cal_df["Dátum"].dt.year
        cal_df["week"] = cal_df["Dátum"].dt.isocalendar().week
        cal_df["dow"] = cal_df["Dátum"].dt.dayofweek
        available_years = sorted(cal_df["year"].unique(), reverse=True)
        if available_years:
            sel_year = st.selectbox("Válassz évet", available_years, index=0, key="cal_year_select")
            df_y = cal_df[cal_df["year"] == sel_year].copy()
            daily_sums = df_y.groupby(["week", "dow"])["dist_km"].sum().reset_index()
            heatmap_data = daily_sums.pivot(index="dow", columns="week", values="dist_km").fillna(0)
            heatmap_data = heatmap_data.reindex(index=range(7), fill_value=0).reindex(columns=range(1, 54), fill_value=0)
            fig_cal = px.imshow(
                heatmap_data,
                labels=dict(x="Hét", y="Nap", color="km"),
                x=heatmap_data.columns,
                y=["Hétfő", "Kedd", "Szerda", "Csütörtök", "Péntek", "Szombat", "Vasárnap"],
                color_continuous_scale="Greens",
                aspect="auto",
                title=f"{sel_year} – Futott km naponta",
            )
            fig_cal.update_xaxes(side="top", dtick=2)
            fig_cal.update_layout(height=300, margin=dict(l=0, r=0, t=50, b=0))
            fig_cal.update_traces(xgap=2, ygap=2)
            st.plotly_chart(fig_cal, use_container_width=True)

    # --- Intenzitás megoszlás
    st.divider()
    st.subheader("⚡ Intenzitás megoszlás heti szinten")
    int_base = d.copy()
    if "Tevékenység típusa" in int_base.columns:
        int_base = int_base[int_base["Tevékenység típusa"].astype(str).str.contains("Fut", na=False)].copy()

    if int_base.empty:
        st.info("Nincs futás adat az intenzitás bontáshoz.")
    else:
        int_base["hr_pct"] = np.where(
            int_base["hr_num"].notna() & (hrmax > 0),
            int_base["hr_num"] / float(hrmax),
            np.nan,
        )
        int_base["hr_zone"] = int_base["hr_pct"].apply(hr_zone_from_pct)
        int_base["rt"] = int_base[run_type_col].astype(str) if run_type_col else "unknown"

        w_idx = int_base.set_index("Dátum")
        weekly_km = w_idx.resample("W-MON")["dist_km"].sum().rename("week_km").reset_index()

        rt_week = (
            w_idx.resample("W-MON")["rt"].value_counts().rename("count")
            .reset_index().rename(columns={"Dátum": "week"})
        )
        rt_pivot = rt_week.pivot_table(index="week", columns="rt", values="count", fill_value=0).reset_index()
        rt_cols = [c for c in rt_pivot.columns if c != "week"]
        rt_pivot["total"] = rt_pivot[rt_cols].sum(axis=1)
        for c in rt_cols:
            rt_pivot[c] = np.where(rt_pivot["total"] > 0, rt_pivot[c] / rt_pivot["total"] * 100.0, 0.0)

        hz_week = (
            w_idx.resample("W-MON")["hr_zone"].value_counts().rename("count")
            .reset_index().rename(columns={"Dátum": "week"})
        )
        hz_pivot = hz_week.pivot_table(index="week", columns="hr_zone", values="count", fill_value=0).reset_index()
        hz_cols = [c for c in ["Z1", "Z2", "Z3", "Z4", "Z5"] if c in hz_pivot.columns]
        hz_pivot["total"] = hz_pivot[hz_cols].sum(axis=1) if hz_cols else 0
        for c in hz_cols:
            hz_pivot[c] = np.where(hz_pivot["total"] > 0, hz_pivot[c] / hz_pivot["total"] * 100.0, 0.0)

        cL, cR = st.columns(2)
        with cL:
            st.markdown("#### 🧩 Easy / Tempo / Race / Interval arány (heti %)")
            show_rts = [c for c in ["easy", "tempo", "race", "interval"] if c in rt_cols]
            if run_type_col and show_rts:
                rt_long = rt_pivot[["week"] + show_rts].melt("week", var_name="Edzés típusa", value_name="Percent")
                rt_long["Edzés típusa"] = rt_long["Edzés típusa"].apply(run_type_hu)
                st.plotly_chart(
                    px.bar(rt_long, x="week", y="Percent", color="Edzés típusa", barmode="stack", labels={"week": "Hét", "Percent": "%"}),
                    use_container_width=True,
                )
            else:
                st.info("Edzés típus hiányzik / nincs felismerve.")

        with cR:
            st.markdown("#### ❤️ HR zóna megoszlás (heti %)")
            if int_base["hr_zone"].notna().sum() >= 5 and hz_cols:
                hz_long = hz_pivot[["week"] + hz_cols].melt("week", var_name="Zone", value_name="Percent")
                st.plotly_chart(
                    px.bar(hz_long, x="week", y="Percent", color="Zone", barmode="stack", labels={"week": "Hét", "Percent": "%"}),
                    use_container_width=True,
                )
            else:
                st.info("Kevés / hiányzó pulzus adat.")

        st.divider()
        st.markdown("#### 🔎 Terhelés vs Intenzitás (heti km + Z4/Z5 arány)")
        if hz_cols:
            hz_pivot["hi_intensity_pct"] = hz_pivot.get("Z4", 0) + hz_pivot.get("Z5", 0)
            combo = weekly_km.rename(columns={"Dátum": "week"}).merge(hz_pivot[["week", "hi_intensity_pct"]], on="week", how="left")
            combo["hi_intensity_pct"] = combo["hi_intensity_pct"].fillna(0.0)
            st.plotly_chart(
                px.scatter(combo, x="week_km", y="hi_intensity_pct", hover_data=["week"], labels={"week_km": "Heti km", "hi_intensity_pct": "Z4+Z5 %"}),
                use_container_width=True,
            )
            if len(combo.dropna(subset=["week_km"])) >= 4:
                last = combo.iloc[-1]
                msg_txt = f"Utolsó hét: **{last['week_km']:.1f} km**, magas intenzitás (Z4+Z5): **{last['hi_intensity_pct']:.0f}%**."
                if last["hi_intensity_pct"] >= 30:
                    st.warning(f"🟠 Sok a magas intenzitás (Z4+Z5) → ez felnyomhatja a Fatigue_score-t.\n\n{msg_txt}")
                else:
                    st.success(f"🟢 A magas intenzitás arány nem extrém.\n\n{msg_txt}")

    # --- Technika idősor
    st.divider()
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("📈 Technika_index időben")
        if "Technika_index" in view.columns and view["Technika_index"].notna().sum() >= 3:
            fig = px.scatter(
                view.dropna(subset=["Technika_index"]),
                x="Dátum",
                y="Technika_index",
                color="Edzés típusa" if "Edzés típusa" in view.columns else None,
                symbol="Terep" if "Terep" in view.columns else None,
                hover_data=[c for c in ["Cím", "Átlagos tempó", fatigue_col, fatigue_type_col, "Terep"] if c and c in view.columns],
                opacity=0.75,
            )
            view2 = view[["Dátum", "Technika_index"]].dropna().sort_values("Dátum").copy()
            if len(view2) >= 10:
                view2["roll30"] = view2["Technika_index"].rolling(window=30, min_periods=10).mean()
                for tr in px.line(view2, x="Dátum", y="roll30").data:
                    fig.add_trace(tr)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nincs elég Technika_index adat az idősorhoz.")

    with right:
        st.subheader("🗺️ Terep megoszlás (szűrve)")
        if slope_col and len(view[slope_col].dropna()) > 0:
            cnt = view[slope_col].apply(slope_bucket_hu).value_counts().reset_index()
            cnt.columns = ["Terep", "db"]
            st.plotly_chart(px.bar(cnt, x="Terep", y="db"), use_container_width=True)
        else:
            st.info("Nincs slope_bucket adat.")

    # --- Technika vs Fáradás kvadráns
    st.divider()
    cA, cB = st.columns(2)
    with cA:
        st.subheader("🧭 Technika vs Fáradás (kvadráns)")
        if (
            "Technika_index" in view.columns
            and fatigue_col
            and view["Technika_index"].notna().sum() >= 10
            and view[fatigue_col].notna().sum() >= 10
        ):
            dd = view.dropna(subset=["Technika_index", fatigue_col]).copy()
            tech_med = float(np.nanmedian(dd["Technika_index"]))
            fat_med = float(np.nanmedian(dd[fatigue_col]))
            fig2 = px.scatter(
                dd,
                x="Technika_index",
                y=fatigue_col,
                color="Edzés típusa" if "Edzés típusa" in dd.columns else None,
                symbol="Terep" if "Terep" in dd.columns else None,
                hover_data=[c for c in ["Dátum", "Cím", "Átlagos tempó", fatigue_type_col, "Terep"] if c and c in dd.columns],
                opacity=0.75,
            )
            fig2.add_vline(x=tech_med)
            fig2.add_hline(y=fat_med)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"Medián határok: Technika {tech_med:.1f}, Fatigue {fat_med:.1f}")
        else:
            st.info("Kevés Technika/Fatigue adat a kvadránshoz.")

    # --- Terhelés vs Technika (heti trend)
    with cB:
        st.subheader("🧭 Terhelés vs Technika (heti trend)")
        if "Technika_index" in d.columns:
            w_tt = d.dropna(subset=["Dátum", "Technika_index"]).copy()
            w_tt["week_start"] = w_tt["Dátum"].dt.to_period("W").dt.start_time
            w_tt["week"] = w_tt["week_start"].dt.strftime("%Y-%m-%d")

            load_col = None
            load_label_wt = None
            if "dist_km" in w_tt.columns and w_tt["dist_km"].notna().any():
                load_col = "dist_km"
                load_label_wt = "Heti táv (km)"
            elif "Idő" in w_tt.columns:
                w_tt["_load"] = w_tt["Idő"].apply(_to_minutes)
                load_col = "_load"
                load_label_wt = "Heti idő (perc)"

            if load_col:
                wkly_tt = (
                    w_tt.groupby(["week_start", "week"], as_index=False)
                    .agg(load_sum=(load_col, "sum"), tech_mean=("Technika_index", "mean"), runs=("Technika_index", "count"))
                    .sort_values("week_start")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=["load_sum", "tech_mean"])
                )
                if len(wkly_tt) >= 6:
                    last6 = wkly_tt.tail(6).copy()
                    x_arr = np.arange(len(last6), dtype=float)
                    y_load = last6["load_sum"].to_numpy(dtype=float)
                    y_tech = last6["tech_mean"].to_numpy(dtype=float)

                    if np.nanstd(y_load) > 1e-9 and np.nanstd(y_tech) > 1e-9:
                        tech_tr = np.polyfit(x_arr, y_tech, 1)[0]
                        load_tr = np.polyfit(x_arr, y_load, 1)[0]
                        if load_tr > 0 and tech_tr < 0:
                            verdict = "🔴 Terhelés nő, technika romlik → túlterhelés gyanú"
                        elif load_tr > 0 and tech_tr < 0.05:
                            verdict = "🟠 Terhelés nő, technika stagnál → határon"
                        elif load_tr > 0 and tech_tr > 0:
                            verdict = "🟢 Terhelés nő, technika javul → adaptáció"
                        else:
                            verdict = "ℹ️ Nincs egyértelmű trend"
                        st.caption(verdict)

                    fig_wt = px.scatter(
                        wkly_tt, x="load_sum", y="tech_mean",
                        hover_data=["week", "runs"],
                        labels={"load_sum": load_label_wt, "tech_mean": "Heti átlag Technika_index"},
                    )
                    tmp_wt = wkly_tt[["load_sum", "tech_mean"]].dropna()
                    if len(tmp_wt) >= 3 and tmp_wt["load_sum"].nunique() >= 2:
                        xfit = tmp_wt["load_sum"].to_numpy(dtype=float)
                        yfit = tmp_wt["tech_mean"].to_numpy(dtype=float)
                        m_f, b_f = np.polyfit(xfit, yfit, 1)
                        xs = np.linspace(xfit.min(), xfit.max(), 40)
                        fig_wt.add_scatter(x=xs, y=m_f * xs + b_f, mode="lines", name="Trend")
                    st.plotly_chart(fig_wt, use_container_width=True)
                else:
                    st.info("Kevés heti adat (min. ~6 hét ajánlott).")
            else:
                st.info("Nincs terhelés adat.")
        else:
            st.info("Nincs Technika_index.")


# =========================================================
# TAB: UTOLSÓ FUTÁS
# =========================================================
with tab_last:
    st.subheader("🔎 Utolsó futás elemzése")

    if "Technika_index" not in d.columns:
        st.info("Nincs Technika_index – az utolsó futás technika elemzéséhez számított index kell.")
    else:
        base = d.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
        if len(base) == 0:
            st.info("Nincs elég adat (Dátum + Technika_index).")
        else:
            options = base.tail(60).copy()

            def label_row(r):
                title = r.get("Cím", "") if "Cím" in options.columns else ""
                rt = run_type_hu(r.get("Run_type", "")) if "Run_type" in options.columns else ""
                return f"{r['Dátum'].strftime('%Y-%m-%d %H:%M')} | {rt} | {title}"[:120]

            options["__label"] = options.apply(label_row, axis=1)
            chosen_label = st.selectbox("Futás kiválasztása", options["__label"].tolist(), index=len(options) - 1, key="pick_last")
            last = options.loc[options["__label"] == chosen_label].iloc[0]

            last_type = None
            if run_type_col and run_type_col in base.columns and pd.notna(last.get(run_type_col)):
                last_type = str(last.get(run_type_col)).strip().lower()

            if baseline_mode == "Auto (Edzés típusa szerint)" and last_type in ("easy", "tempo", "race", "interval"):
                baseline_full = get_type_baseline(
                    base_df=base, last_date=last["Dátum"], run_type_col=run_type_col,
                    target_type=last_type, weeks=baseline_weeks, min_runs=baseline_min_runs,
                )
                st.caption(f"Baseline: **{last_type}** futások (hetek: {baseline_weeks}, min: {baseline_min_runs})")
            else:
                baseline_full = get_easy_baseline(base_df=base, last_date=last["Dátum"], weeks=baseline_weeks, min_runs=baseline_min_runs)
                st.caption(f"Baseline: **easy** futások (hetek: {baseline_weeks}, min: {baseline_min_runs})")

            # --- Fatmax & Aerobic decoupling
            st.divider()
            st.subheader("🔥 Fatmax & Aerobic decoupling (proxy)")

            hr_col = find_col(base, "Átlagos pulzusszám", "Átlagos pulzusszám ")
            pace_col_f = find_col(base, "Átlagos tempó")
            pwr_col = find_col(base, "Átl. teljesítmény", "Átlagos teljesítmény", "Average Power", "Avg Power")
            pwr_max_col = find_col(base, "Max. teljesítmény", "Maximum Power", "Max Power")

            if hr_col is None or pace_col_f is None:
                st.info("Fatmax-hoz kell: **Átlagos pulzusszám** és **Átlagos tempó** oszlop.")
            else:
                fat_base = baseline_full.copy() if baseline_full is not None and len(baseline_full) > 0 else base.copy()
                fatmax = estimate_fatmax_from_runs(fat_base, hrmax, hr_col, pace_col_f, pwr_col, min_points=12)
                dec = aerobic_decoupling_proxy(last, baseline_full, hr_col, pace_col_f, pwr_col)

                k1, k2, k3, k4 = st.columns(4)
                if fatmax is None:
                    k1.metric("Fatmax HR", "—"); k2.metric("Fatmax tempó", "—")
                    k3.metric("Fatmax power", "—"); k4.metric("Decoupling (proxy)", "—")
                    st.caption("Kevés adat a Fatmax becsléshez (kell ~12+ releváns futás).")
                else:
                    k1.metric("Fatmax HR (becslés)", f"{fatmax['hr_fatmax']:.0f} bpm", delta=f"HR% bin: {fatmax['best_bin']}")
                    k2.metric("Fatmax tempó (becslés)", f"{fatmax['pace_fatmax']}/km", delta=f"N={fatmax['n_points']}")
                    k3.metric("Fatmax power (becslés)", f"{fatmax['pwr_fatmax']:.0f} W" if pd.notna(fatmax["pwr_fatmax"]) else "—", delta=f"EF: {fatmax['ef_col']}")
                    k4.metric("Aerobic decoupling (proxy)", f"{dec['delta_pct']:+.1f}%" if dec else "—", delta=dec["ef"] if dec else None)

                    # Power arány
                    last_pwr = to_float_series(pd.Series([last.get(pwr_col)])).iloc[0] if pwr_col and pwr_col in base.columns else np.nan
                    last_pwr_max = to_float_series(pd.Series([last.get(pwr_max_col)])).iloc[0] if pwr_max_col and pwr_max_col in base.columns else np.nan
                    if pd.notna(last_pwr) and pd.notna(last_pwr_max) and last_pwr > 0:
                        ratio = float(last_pwr_max / last_pwr)
                        if ratio >= 2.0:
                            st.warning(f"⚠️ Csúcsos terhelés: Max/Avg power = **{ratio:.2f}** → domb/megindulások torzíthatják a technikát.")
                        else:
                            st.success(f"✅ Terhelés eloszlás: Max/Avg power = **{ratio:.2f}** (nem extrém csúcsos).")

                    # EF plot
                    dfp = fatmax["table"].dropna(subset=["HR_pct", fatmax["ef_col"]]).copy()
                    if len(dfp) >= 10:
                        fig_ef = px.scatter(
                            dfp, x="HR_pct", y=fatmax["ef_col"],
                            hover_data=["Dátum", "Run_type"] + (["Cím"] if "Cím" in dfp.columns else []),
                            labels={"HR_pct": "HR%", fatmax["ef_col"]: fatmax["ef_col"]},
                            title="Fatmax becslés: EF az aerob tartományban",
                        )
                        if pd.notna(fatmax.get("hrpct_mid")):
                            fig_ef.add_vline(x=fatmax["hrpct_mid"])
                        st.plotly_chart(fig_ef, use_container_width=True)

                    # Fatmax trend
                    df_tr = fat_base.dropna(subset=["Dátum"]).sort_values("Dátum").copy()
                    if len(df_tr) >= 20:
                        df_tr["_week"] = df_tr["Dátum"].dt.to_period("W").dt.start_time
                        rows = []
                        for wk, g in df_tr.groupby("_week"):
                            fm = estimate_fatmax_from_runs(g, hrmax, hr_col, pace_col_f, pwr_col, min_points=8)
                            if fm:
                                rows.append([wk, fm["hr_fatmax"], fm["pace_fatmax"], fm["pwr_fatmax"]])
                        if len(rows) >= 6:
                            tdf = pd.DataFrame(rows, columns=["week", "fatmax_hr", "fatmax_pace_str", "fatmax_pwr"])
                            tdf["fatmax_pace_sec"] = tdf["fatmax_pace_str"].apply(pace_str_to_sec)
                            st.plotly_chart(
                                px.line(tdf.sort_values("week"), x="week", y="fatmax_pace_sec",
                                        labels={"week": "Hét", "fatmax_pace_sec": "Fatmax tempó (sec/km)"},
                                        title="Fatmax tempó trend (heti becslés)"),
                                use_container_width=True,
                            )

            # --- RES+
            st.markdown("### ⚡ RES+ (Running Economy) – utolsó futás")
            res_available = "RES_plus" in base.columns and base["RES_plus"].notna().any()
            res_last = float(last.get("RES_plus")) if pd.notna(last.get("RES_plus")) else np.nan
            res_base_val = (
                float(np.nanmedian(baseline_full["RES_plus"]))
                if ("RES_plus" in baseline_full.columns and baseline_full["RES_plus"].notna().any())
                else np.nan
            )
            res_delta = (res_last - res_base_val) if (pd.notna(res_last) and pd.notna(res_base_val)) else np.nan
            pavg_last = float(last.get("power_avg_w")) if pd.notna(last.get("power_avg_w")) else np.nan
            pmax_last = float(last.get("power_max_w")) if pd.notna(last.get("power_max_w")) else np.nan
            p_ratio = (
                float(last.get("Power_fatigue_hint"))
                if pd.notna(last.get("Power_fatigue_hint"))
                else ((pmax_last / pavg_last) if (pd.notna(pmax_last) and pd.notna(pavg_last) and pavg_last > 0) else np.nan)
            )

            if not res_available:
                st.info("Nincs RES_plus adat (hiányos power vagy kevés futás).")
            else:
                cR1, cR2, cR3, cR4 = st.columns(4)
                cR1.metric("RES_plus", f"{res_last:.1f}" if pd.notna(res_last) else "—")
                cR2.metric("RES_plus baseline", f"{res_base_val:.1f}" if pd.notna(res_base_val) else "—")
                cR3.metric("Eltérés", f"{res_delta:+.1f}" if pd.notna(res_delta) else "—")
                cR4.metric("Átl. teljesítmény", f"{pavg_last:.0f} W" if pd.notna(pavg_last) else "—")

                if pd.notna(res_delta):
                    if res_delta >= 6:
                        st.write("🟢 **RES+ nagyot javult** → gazdaságosabb futás.")
                    elif res_delta >= 2:
                        st.write("🟢 **RES+ kicsit jobb** → stabilan jó irány.")
                    elif res_delta > -2:
                        st.write("🟡 **RES+ kb. baseline** → normál ingadozás.")
                    elif res_delta > -6:
                        st.write("🟠 **RES+ romlott** → lehet domb/hőség/szél vagy technika/fáradás.")
                    else:
                        st.write("🔴 **RES+ nagyon romlott** → valószínű erős külső tényező vagy fáradás.")

                if pd.notna(p_ratio):
                    if p_ratio >= 1.60:
                        st.write(f"🟠 **Csúcsos terhelés** (Max/Átl power ≈ {p_ratio:.2f}) → torzítás lehetséges.")
                    elif p_ratio >= 1.40:
                        st.write(f"🟡 **Változékony terhelés** (Max/Átl power ≈ {p_ratio:.2f}).")
                    else:
                        st.write(f"🟢 **Egyenletes terhelés** (Max/Átl power ≈ {p_ratio:.2f}).")

                if "RES_plus" in baseline_full.columns and baseline_full["RES_plus"].notna().sum() >= 8 and pd.notna(res_last):
                    tmpb = baseline_full.dropna(subset=["RES_plus"])[["RES_plus"]].copy()
                    tmpb["label"] = "baseline"
                    last_row_res = pd.DataFrame({"RES_plus": [res_last], "label": ["utolsó"]})
                    tmp_res = pd.concat([tmpb, last_row_res], ignore_index=True)
                    st.plotly_chart(
                        px.histogram(tmp_res, x="RES_plus", color="label", barmode="overlay", nbins=18,
                                     title="RES+ eloszlás a baseline-ban + utolsó futás"),
                        use_container_width=True,
                    )

            # --- KPI-k
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Technika_index", f"{float(last['Technika_index']):.1f}")
            fat_val = last.get("Fatigue_score")
            c2.metric("Fatigue_score", f"{float(fat_val):.1f}" if pd.notna(fat_val) else "—")
            c3.metric("Edzés típusa", run_type_hu(last.get("Run_type")) if pd.notna(last.get("Run_type")) else "—")
            pace_v = last.get("Átlagos tempó") if "Átlagos tempó" in base.columns else None
            dist_v = last.get("Távolság") if "Távolság" in base.columns else None
            c4.metric("Tempó / Táv", f"{pace_v} / {dist_v} km" if (pd.notna(pace_v) or pd.notna(dist_v)) else "—")

            if "Cím" in base.columns and pd.notna(last.get("Cím")):
                st.caption(f"**Cím:** {last['Cím']}")
            st.caption(f"Baseline futások száma: **{len(baseline_full)}**")

            if len(baseline_full) < 8:
                st.info("Kevés baseline futás a biztos elemzéshez (ajánlott ≥ 8–10).")
                if "Run_type" in base.columns:
                    cnts = base["Run_type"].value_counts(dropna=False).head(10)
                    st.write("DEBUG – Edzés típus megoszlás:")
                    st.dataframe(cnts.rename("db").reset_index().rename(columns={"index": "Run_type"}), hide_index=True)
                    st.write("DEBUG last_type:", last_type, "| baseline_mode:", baseline_mode)
            else:
                compare_cols = [
                    ("Átl. pedálütem", "Cadence (spm)", "cadence_stability"),
                    ("Átlagos lépéshossz", "Lépéshossz (m)", "higher_better"),
                    ("Átlagos függőleges arány", "Vertical Ratio (%)", "lower_better"),
                    ("Átlagos függőleges oszcilláció", "Vertical Osc (cm)", "lower_better"),
                    ("Átlagos talajérintési idő", "GCT (ms)", "lower_better"),
                    ("Átlagos pulzusszám", "Átlag pulzus", "context"),
                    ("Max. pulzusszám", "Max pulzus", "context"),
                ]
                rows = []
                for col_c, label_c, rule_c in compare_cols:
                    if col_c not in base.columns:
                        continue
                    v = num(last.get(col_c))
                    b_v = med(baseline_full[col_c])
                    if pd.isna(v) or pd.isna(b_v) or b_v == 0:
                        continue
                    delta_pct = (v - b_v) / b_v * 100.0
                    good = (
                        -delta_pct if rule_c == "lower_better"
                        else delta_pct if rule_c == "higher_better"
                        else -abs(delta_pct) if rule_c == "cadence_stability"
                        else 0.0
                    )
                    rows.append([label_c, float(v), float(b_v), float(delta_pct), float(good), rule_c])

                if rows:
                    comp = pd.DataFrame(rows, columns=["Mutató", "Utolsó", "Baseline", "Eltérés_%", "Jó_irány", "rule"])
                    st.markdown("### 📈 Eltérések a baseline-hoz képest")
                    st.plotly_chart(
                        px.bar(comp.sort_values("Jó_irány"), x="Jó_irány", y="Mutató", orientation="h",
                               hover_data=["Utolsó", "Baseline", "Eltérés_%", "rule"]),
                        use_container_width=True,
                    )
                    st.markdown("### 🚩 Gyors jelzések")
                    for _, r in comp.iterrows():
                        if r["rule"] in ("lower_better", "higher_better"):
                            if r["Jó_irány"] < -5:
                                st.write(f"🔴 **{r['Mutató']}** romlott (≈ {r['Eltérés_%']:+.1f}%).")
                            elif r["Jó_irány"] < -2:
                                st.write(f"🟠 **{r['Mutató']}** kicsit romlott (≈ {r['Eltérés_%']:+.1f}%).")
                            else:
                                st.write(f"🟢 **{r['Mutató']}** rendben (≈ {r['Eltérés_%']:+.1f}%).")
                        elif r["rule"] == "cadence_stability":
                            if abs(r["Eltérés_%"]) > 5:
                                st.write(f"🟠 **{r['Mutató']}** eltér a baseline-tól (≈ {r['Eltérés_%']:+.1f}%).")
                            else:
                                st.write(f"🟢 **{r['Mutató']}** stabil (≈ {r['Eltérés_%']:+.1f}%).")

                    with st.expander("📋 Részletes táblázat"):
                        st.dataframe(comp.drop(columns=["Jó_irány"]), use_container_width=True, hide_index=True)
                else:
                    st.info("Nincs elég összehasonlítható metrika.")


# =========================================================
# TAB: WARNING
# =========================================================
with tab_warn:
    st.subheader("🚦 Warning rendszer (easy futások alapján)")

    if "Technika_index" not in d.columns or fatigue_col is None:
        st.info("Warning-hoz kell Technika_index és Fatigue_score.")
    else:
        with st.expander("Beállítások", expanded=False):
            colA, colB, colC = st.columns(3)
            tech_red = colA.slider("Tech küszöb (PIROS)", 0, 100, 35, key="wr_tech_red")
            tech_yellow = colA.slider("Tech küszöb (SÁRGA)", 0, 100, 40, key="wr_tech_yellow")
            fat_yellow = colB.slider("Fatigue küszöb (SÁRGA)", 0, 100, 60, key="wr_fat_yellow")
            fat_red = colB.slider("Fatigue küszöb (PIROS)", 0, 100, 55, key="wr_fat_red")
            n_red = colC.slider("N easy futás (PIROS ablak)", 3, 12, 3, key="wr_n_red")
            need_red = colC.slider("Minimum találat (PIROS)", 1, 12, 2, key="wr_need_red")
            n_yellow = colC.slider("N easy futás (SÁRGA ablak)", 3, 20, 5, key="wr_n_yellow")
            need_yellow = colC.slider("Minimum találat (SÁRGA)", 1, 20, 2, key="wr_need_yellow")

        warn_base = d.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
        easy_w = (
            warn_base[warn_base[run_type_col] == "easy"].copy()
            if run_type_col
            else warn_base.copy()
        )
        if len(easy_w) > 0:
            last_day_w = easy_w["Dátum"].max()
            start_w = last_day_w - pd.Timedelta(weeks=baseline_weeks)
            easy_w = easy_w[easy_w["Dátum"] >= start_w].copy()
            if len(easy_w) < baseline_min_runs:
                easy_w = easy_w.tail(baseline_min_runs).copy()

        easy_f = easy_w.dropna(subset=[fatigue_col]).copy()
        if len(easy_f) < 5:
            st.info("Nincs elég easy + Fatigue_score adat (legalább ~5 futás).")
        else:
            last_red_df = easy_f.tail(n_red).copy()
            last_yellow_df = easy_f.tail(n_yellow).copy()
            last_red_df["hit_red"] = (last_red_df["Technika_index"] < tech_red) & (last_red_df[fatigue_col] > fat_red)
            last_yellow_df["hit_yellow"] = (last_yellow_df["Technika_index"] < tech_yellow) | (last_yellow_df[fatigue_col] > fat_yellow)
            red_hits = int(last_red_df["hit_red"].sum())
            yellow_hits = int(last_yellow_df["hit_yellow"].sum())

            if red_hits >= need_red:
                status_w = "🔴 PIROS"
                reason_w = f"Utolsó {n_red} easy futásból {red_hits} találat: Tech < {tech_red} ÉS Fatigue > {fat_red}."
                st.error(f"{status_w} — {reason_w}")
            elif yellow_hits >= need_yellow:
                status_w = "🟠 SÁRGA"
                reason_w = f"Utolsó {n_yellow} easy futásból {yellow_hits} találat: Tech < {tech_yellow} VAGY Fatigue > {fat_yellow}."
                st.warning(f"{status_w} — {reason_w}")
            else:
                status_w = "🟢 ZÖLD"
                st.success(f"{status_w} — Stabil easy technika / fáradás kontrollált.")

            c1_w, c2_w = st.columns(2)
            c1_w.metric("PIROS találatok", f"{red_hits}/{n_red}")
            c2_w.metric("SÁRGA találatok", f"{yellow_hits}/{n_yellow}")

            show_w = easy_f.tail(max(n_yellow, n_red)).copy()
            show_w["red_hit"] = (show_w["Technika_index"] < tech_red) & (show_w[fatigue_col] > fat_red)
            show_w["yellow_hit"] = (show_w["Technika_index"] < tech_yellow) | (show_w[fatigue_col] > fat_yellow)

            st.plotly_chart(
                px.scatter(show_w, x="Dátum", y="Technika_index", color="yellow_hit", symbol="red_hit",
                           hover_data=["Cím"] if "Cím" in show_w.columns else None),
                use_container_width=True,
            )

            with st.expander("📋 Részletek"):
                cols_w = ["Dátum", "Technika_index", fatigue_col, "red_hit", "yellow_hit"]
                if "Cím" in show_w.columns:
                    cols_w.append("Cím")
                st.dataframe(show_w.sort_values("Dátum", ascending=False)[cols_w], use_container_width=True, hide_index=True)


# =========================================================
# TAB: READINESS
# =========================================================
with tab_ready:
    st.subheader("🏁 Verseny-előrejelzés (Readiness) – 14 napos ablak")

    if run_type_col is None or fatigue_col is None or "Technika_index" not in d.columns:
        st.info("Readiness-hez kell Edzés típusa + Fatigue_score + Technika_index.")
    else:
        ready_base = d.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
        easy_r = ready_base[ready_base[run_type_col] == "easy"].dropna(subset=[fatigue_col]).copy()

        if len(easy_r) < 10:
            st.info("Nincs elég easy + Fatigue_score adat a readiness-hez.")
        else:
            default_date = ready_base["Dátum"].max().date()
            race_date = st.date_input("Verseny dátuma", value=default_date, key="race_date")
            window_days = st.slider("Ablak (nap)", 7, 28, 14, key="race_window")

            start_r = pd.Timestamp(race_date) - pd.Timedelta(days=window_days)
            end_r = pd.Timestamp(race_date)
            w_r = easy_r[(easy_r["Dátum"] >= start_r) & (easy_r["Dátum"] <= end_r)].copy()

            if len(w_r) < 3:
                st.warning(f"Az utolsó {window_days} napban kevés easy futás van ({len(w_r)}).")
            else:
                tech_mean_r = float(w_r["Technika_index"].mean())
                fat_mean_r = float(w_r[fatigue_col].mean())

                w2_r = w_r.sort_values("Dátum")[["Dátum", "Technika_index"]].dropna().copy()
                x_r = (w2_r["Dátum"] - w2_r["Dátum"].min()).dt.total_seconds().to_numpy()
                y_r = w2_r["Technika_index"].to_numpy()
                slope_per_day = 0.0
                if len(w2_r) >= 3 and np.nanstd(x_r) > 0:
                    slope_per_day = float(np.polyfit(x_r, y_r, 1)[0]) * 86400.0

                red_hits_r = int(((w_r["Technika_index"] < 35) & (w_r[fatigue_col] > 55)).sum())
                tech_p25, tech_p75 = np.nanpercentile(easy_r["Technika_index"], [25, 75])
                fat_p25, fat_p75 = np.nanpercentile(easy_r[fatigue_col], [25, 75])

                tech_score_r = float(np.clip(100 * (tech_mean_r - tech_p25) / (tech_p75 - tech_p25 + 1e-9), 0, 100))
                fat_score_r = float(np.clip(100 * (fat_p75 - fat_mean_r) / (fat_p75 - fat_p25 + 1e-9), 0, 100))
                trend_score_r = float(np.clip(50 + 20 * slope_per_day, 0, 100))
                penalty_r = min(30, red_hits_r * 10)
                readiness = float(np.clip(0.50 * tech_score_r + 0.30 * fat_score_r + 0.20 * trend_score_r - penalty_r, 0, 100))

                status_r = "🟢 ZÖLD" if readiness >= 60 and red_hits_r == 0 else ("🔴 PIROS" if readiness < 40 or red_hits_r >= 2 else "🟠 SÁRGA")

                c1_r, c2_r, c3_r, c4_r = st.columns(4)
                c1_r.metric("Readiness_score", f"{readiness:.0f}")
                c2_r.metric("Tech (easy) átlag", f"{tech_mean_r:.1f}")
                c3_r.metric("Fatigue (easy) átlag", f"{fat_mean_r:.1f}")
                c4_r.metric("Tech trend (pont/nap)", f"{slope_per_day:+.2f}")

                if status_r.startswith("🟢"):
                    st.success("🟢 Jó verseny-készenlét (taper + stabil easy).")
                elif status_r.startswith("🟠"):
                    st.warning("🟠 Közepes készenlét (még van fáradás / instabil technika).")
                else:
                    st.error("🔴 Nem ideális (fáradás magas és/vagy technika szétesik easy-n is).")

                st.plotly_chart(
                    px.line(w_r.sort_values("Dátum"), x="Dátum", y=["Technika_index", fatigue_col]),
                    use_container_width=True,
                )

                with st.expander("📋 Ablakban lévő futások"):
                    show_cols_r = ["Dátum", "Technika_index", fatigue_col]
                    if "Cím" in w_r.columns:
                        show_cols_r.append("Cím")
                    st.dataframe(w_r.sort_values("Dátum", ascending=False)[show_cols_r], use_container_width=True, hide_index=True)


# =========================================================
# TAB: ADATOK
# =========================================================
with tab_data:
    st.subheader("📄 Adatok (szűrve)")
    st.caption("A szűrt futások teljes adattáblája. Az elemzésekhez elég az első 4 tab.")
    st.dataframe(view, use_container_width=True, hide_index=True, height=520)
