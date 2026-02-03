import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import csv
import os
from datetime import datetime

st.set_page_config(page_title="Garmin Futás Dashboard", layout="wide")
# =========================================================
# 📱 MOBIL NÉZET (kapcsoló + UI finomhangolás)
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
        /* kisebb margók mobilon */
        .block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; padding-left: 0.8rem; padding-right: 0.8rem; }

        /* sidebar kicsit kompaktabb */
        section[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }

        /* metric kártyák kompaktabbak */
        div[data-testid="stMetric"] {
            padding: 0.6rem 0.8rem;
            border-radius: 12px;
        }
        div[data-testid="stMetric"] label { font-size: 0.85rem !important; }
        div[data-testid="stMetric"] div { font-size: 1.4rem !important; }

        /* dataframe ne legyen óriás */
        div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

        /* fejlécek kicsit kisebbek */
        h1 { font-size: 1.6rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }

        </style>
        """,
        unsafe_allow_html=True
    )

inject_mobile_css(MOBILE)

def metric_row(mobile: bool, items):
    """
    items: list of tuples (label, value, delta_optional)
    """
    if mobile:
        cols = st.columns(2)
        for i, it in enumerate(items):
            label = it[0]
            value = it[1]
            delta = it[2] if len(it) > 2 else None
            with cols[i % 2]:
                st.metric(label, value, delta=delta)
    else:
        cols = st.columns(len(items))
        for col, it in zip(cols, items):
            label = it[0]
            value = it[1]
            delta = it[2] if len(it) > 2 else None
            col.metric(label, value, delta=delta)


# =========================================================
# 0) AUTH (jelszavas)
# =========================================================
def require_password():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
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
# 1) SEGÉDFÜGGVÉNYEK
# =========================================================
def to_float_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"--": np.nan, "": np.nan, "None": np.nan})
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)  # HU tizedes -> pont
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
    except:
        return np.nan

def robust_z(x: pd.Series, ref: pd.Series) -> pd.Series:
    refv = ref.to_numpy(dtype=float)
    med = np.nanmedian(refv)
    mad = np.nanmedian(np.abs(refv - med)) + 1e-9
    return (x - med) / (1.4826 * mad)

def safe_col(df: pd.DataFrame, name: str):
    return name if name in df.columns else None

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

def get_easy_baseline(base_df: pd.DataFrame, last_date: pd.Timestamp, weeks: int, min_runs: int) -> pd.DataFrame:
    if "Run_type" in base_df.columns:
        easy = base_df[base_df["Run_type"] == "easy"].copy()
    else:
        easy = base_df.copy()

    easy = easy.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")

    if pd.isna(last_date):
        return easy.tail(min_runs).copy()

    start = last_date - pd.Timedelta(weeks=weeks)
    w = easy[(easy["Dátum"] >= start) & (easy["Dátum"] <= last_date)].copy()

    if len(w) < min_runs:
        w = easy.tail(min_runs).copy()

    return w

def num(v):
    return pd.to_numeric(pd.Series([v]).astype(str).str.replace(",", "", regex=False), errors="coerce").iloc[0]

def med(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce").median()
def daily_coach_summary(base_all: pd.DataFrame,
                        run_type_col: str | None,
                        fatigue_col: str | None,
                        baseline_weeks: int,
                        baseline_min_runs: int) -> tuple[str, str]:
    """
    Visszaad: (status_emoji, 1 mondatos coach üzenet)
    Logika: legutóbbi easy futás összevetése easy baseline-nal + fáradás trend.
    """
    if base_all is None or len(base_all) == 0:
        return "ℹ️", "Nincs elég adat a napi összképhez."

    # csak technikás futások
    b = base_all.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
    if len(b) < 5:
        return "ℹ️", "Nincs elég technika adat (legalább ~5 futás kell)."

    # easy futások
    if run_type_col and run_type_col in b.columns:
        easy = b[b[run_type_col] == "easy"].copy()
    else:
        easy = b.copy()

    easy = easy.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
    if len(easy) < 5:
        return "ℹ️", "Kevés easy futás → a napi összkép bizonytalan."

    last_easy = easy.iloc[-1]
    baseline_full = get_easy_baseline(
        base_df=b,
        last_date=last_easy["Dátum"],
        weeks=baseline_weeks,
        min_runs=baseline_min_runs
    )

    if len(baseline_full) < 10:
        return "ℹ️", "Kevés easy baseline (min ~10) → még gyűjts adatot 1–2 hétig."

    # baseline mediánok
    tech_b = float(np.nanmedian(baseline_full["Technika_index"]))
    tech_last = float(last_easy["Technika_index"])
    tech_delta = tech_last - tech_b

    # fáradás (ha van)
    fat_last = None
    fat_b = None
    if fatigue_col and fatigue_col in b.columns and baseline_full[fatigue_col].notna().sum() >= 8:
        fat_last = float(last_easy.get(fatigue_col)) if pd.notna(last_easy.get(fatigue_col)) else None
        fat_b = float(np.nanmedian(baseline_full[fatigue_col]))
    # trend: utolsó 7 easy technika
    tail7 = easy.tail(7)
    tech_trend = float(tail7["Technika_index"].iloc[-1] - tail7["Technika_index"].iloc[0]) if len(tail7) >= 4 else 0.0

    # --- döntés logika (egyszerű, de stabil)
    # küszöbök (tuningolható)
    tech_bad = tech_delta < -5
    tech_warn = (-5 <= tech_delta < -2)

    fat_bad = (fat_last is not None and fat_last >= 60)
    fat_warn = (fat_last is not None and 45 <= fat_last < 60)

    # 1 mondatos javaslat generálás
    if tech_bad and (fat_bad or fat_warn):
        status = "🔴"
        msg = f"A technika az easy baseline alatt van (≈{tech_delta:+.1f}), és fáradtabb vagy → holnap inkább pihenő / rövid laza futás + 6×20 mp könnyű repülő."
    elif tech_bad:
        status = "🟠"
        msg = f"A technika az easy baseline alatt van (≈{tech_delta:+.1f}) → holnap legyen könnyebb nap: rövidebb easy, fókusz: ritmus + laza talajfogás."
    elif tech_warn or (fat_warn and not tech_bad):
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

    # trend finomhangolás (ha esik)
    if tech_trend <= -6 and status == "🟢":
        status = "🟠"
        msg = "Az utolsó napokban csúszik a technika trend → holnap inkább könnyebb nap és több regeneráció."

    return status, msg
def duration_to_seconds(x):
    """
    Garmin export: '01:52:42' vagy '00:00:27.7' vagy '--'
    -> másodperc (float)
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in ("", "--", "None", "nan"):
        return np.nan

    # tizedes levágás (00:00:27.7 -> 00:00:27)
    s = s.replace(",", ".")  # ha valahol vessző van
    s_main = s.split(".")[0]

    parts = s_main.split(":")
    try:
        parts = [int(p) for p in parts]
    except:
        return np.nan

    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    return np.nan
def hr_zone_from_pct(pct: float) -> str:
    """
    Egyszerű 5 zónás felosztás HRmax %-ból.
    Z1: <60%
    Z2: 60-70%
    Z3: 70-80%
    Z4: 80-90%
    Z5: >=90%
    """
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

    

# =========================================================
# 2) BEOLVASÁS (upload-only)
# =========================================================
st.sidebar.header("Adatforrás")
uploaded = st.sidebar.file_uploader("Tölts fel Garmin exportot (XLSX ajánlott)", type=["xlsx", "csv"])

if uploaded is None:
    st.title("🏃 Garmin Futás Dashboard")
    st.info("Tölts fel egy XLSX vagy CSV Garmin exportot. (Nem mentjük el az adatokat.)")
    st.stop()

@st.cache_data(show_spinner=False)
def load_any(file) -> pd.DataFrame:
    name = getattr(file, "name", "").lower()

    # ---- XLSX
    if name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")

    # ---- CSV (robosztus: csv.reader + encoding fallback)
    raw = file.getvalue()

    # 1) encoding próbák (Garmin CSV néha nem utf-8)
    text = None
    for enc in ["utf-8-sig", "utf-8", "cp1250", "latin1"]:
        try:
            text = raw.decode(enc)
            break
        except:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")

    lines = text.strip().splitlines()
    if not lines:
        return pd.DataFrame()

    # 2) header + sorok (a te scripted logikád)
    header = lines[0].split(",")
    data_lines = lines[1:]

    data = []
    for raw_data in data_lines:
        raw_data = raw_data.replace('""', '"')
        if raw_data.startswith('"'):
            raw_data = raw_data[1:]
        reader = csv.reader([raw_data], delimiter=",", quotechar='"')
        row = next(reader)
        data.append(row)

    df = pd.DataFrame(data, columns=header)

    # 3) oszlopnevek takarítása (BOM / whitespace)
    df.columns = pd.Index(df.columns).astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    return df



df0 = load_any(uploaded)
def fix_mojibake_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ha ilyen "DĂˇtum", "TevĂ©kenysĂ©g..." jellegű oszlopok vannak,
    # akkor valószínű latin1->utf8 félreértelmezés történt.
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

df0 = fix_mojibake_columns(df0)

df = df0.copy()

# =========================================================
# 3) NORMALIZÁLÁS / ELŐKÉSZÍTÉS
# =========================================================
# Dátum (robosztus parsing CSV-hez is)
df = df0.copy()

# --- Dátum oszlop felderítés + parse (CSV-hez is)
date_candidates = [
    c for c in df.columns
    if "dátum" in c.lower() or "datum" in c.lower() or "date" in c.lower()
]

if date_candidates:
    date_col = date_candidates[0]

    s = (
        df[date_col]
        .astype(str)
        .str.strip()
        .replace({"--": np.nan, "": np.nan, "None": np.nan})
    )

    # 1️⃣ első próbálkozás: fix ISO formátum (CSV-dhez ez a jó)
    dt = pd.to_datetime(
        s,
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S"
    )

    # 2️⃣ fallback: ha mégis eltérő formátum lenne
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(s, errors="coerce")

    df["Dátum"] = dt

else:
    df["Dátum"] = pd.NaT





if "Tevékenység típusa" in df.columns:
    mask_run = df["Tevékenység típusa"].astype(str).str.contains("Fut", na=False)
else:
    mask_run = pd.Series(True, index=df.index)

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

for src, dst in NUM_MAP.items():
    df[dst] = to_float_series(df[src]) if src in df.columns else np.nan

df["pace_sec_km"] = df["Átlagos tempó"].apply(pace_to_sec_per_km) if "Átlagos tempó" in df.columns else np.nan
df["speed_mps"] = np.where(df["pace_sec_km"].notna() & (df["pace_sec_km"] > 0), 1000.0 / df["pace_sec_km"], np.nan)

# =========================================================
# 4) SLOPE FEATURE-ÖK (szint + lejtő)
# =========================================================
df["up_m_per_km"] = np.where(df["dist_km"].notna() & (df["dist_km"] > 0), df["asc_m"] / df["dist_km"], np.nan)
df["down_m_per_km"] = np.where(df["dist_km"].notna() & (df["dist_km"] > 0), df["des_m"] / df["dist_km"], np.nan)
df["net_elev_m"] = df["asc_m"] - df["des_m"]
df["slope_bucket"] = [slope_bucket_row(u, dwn) for u, dwn in zip(df["up_m_per_km"], df["down_m_per_km"])]

# =========================================================
# 5) RUN_TYPE (cím + tempó fallback)
# =========================================================
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
        if ps >= p40:
            return "easy"
        if ps >= p80:
            return "tempo"
        return "race"

    df["Run_type"] = df.apply(classify_from_pace, axis=1)

# =========================================================
# 6) TECHNIKA_INDEX (slope-aware + speed bin)
# =========================================================
df["Technika_index"] = np.nan
tech_base = df[mask_run & df["speed_mps"].notna()].copy()

if len(tech_base) >= 20:
    tech_base["speed_bin"] = pd.qcut(tech_base["speed_mps"], q=8, duplicates="drop")
    for col in ["skill_vr", "skill_gct", "skill_vo", "skill_cad", "skill_stride"]:
        tech_base[col] = np.nan

    grouped = tech_base.groupby(["speed_bin", "slope_bucket"], dropna=False)

    def fill_skills(g: pd.DataFrame):
        idx = g.index
        if g["vr_num"].notna().sum() >= 8:
            tech_base.loc[idx, "skill_vr"] = -robust_z(g["vr_num"], g["vr_num"])
        if g["gct_num"].notna().sum() >= 8:
            tech_base.loc[idx, "skill_gct"] = -robust_z(g["gct_num"], g["gct_num"])
        if g["vo_num"].notna().sum() >= 8:
            tech_base.loc[idx, "skill_vo"] = -robust_z(g["vo_num"], g["vo_num"])
        if g["cad_num"].notna().sum() >= 8:
            z = robust_z(g["cad_num"], g["cad_num"])
            tech_base.loc[idx, "skill_cad"] = -np.abs(z)
        if g["stride_num"].notna().sum() >= 8:
            tech_base.loc[idx, "skill_stride"] = robust_z(g["stride_num"], g["stride_num"])

    for _, g in grouped:
        if len(g) >= 12:
            fill_skills(g)

    still_nan = tech_base["skill_vr"].isna() & tech_base["skill_gct"].isna() & tech_base["skill_vo"].isna() & tech_base["skill_cad"].isna()
    if still_nan.any():
        for _, g in tech_base[still_nan].groupby("speed_bin"):
            fill_skills(g)

    raw = (
        0.25 * tech_base["skill_vr"] +
        0.30 * tech_base["skill_gct"] +
        0.15 * tech_base["skill_vo"] +
        0.20 * tech_base["skill_cad"] +
        0.10 * tech_base["skill_stride"]
    )
    p5, p95 = np.nanpercentile(raw, 5), np.nanpercentile(raw, 95)
    tech_base["Technika_index"] = (100 * (raw - p5) / (p95 - p5 + 1e-9)).clip(0, 100)
    df.loc[tech_base.index, "Technika_index"] = tech_base["Technika_index"].values

# =========================================================
# 7) FATIGUE_SCORE (slope-aware baseline)
# =========================================================
df["Fatigue_score"] = np.nan
df["Fatigue_flag"] = np.nan
df["Fatigue_type"] = np.nan

fat = df[mask_run & df["Technika_index"].notna()].copy()
fat["hr_per_pace"] = np.where(
    fat["hr_num"].notna() & fat["pace_sec_km"].notna() & (fat["pace_sec_km"] > 0),
    fat["hr_num"] / fat["pace_sec_km"],
    np.nan
)

def baseline_for_row(row, fat_df):
    base = fat_df[fat_df["Run_type"] == "easy"].copy()
    if len(base) >= 30:
        sb = row.get("slope_bucket", np.nan)
        b2 = base[base["slope_bucket"] == sb]
        if len(b2) >= 20:
            return b2
        return base
    if len(base) >= 10:
        return base
    return fat_df

if len(fat) >= 20:
    for c in ["fatigue_gct", "fatigue_vr", "fatigue_cad", "fatigue_hr"]:
        fat[c] = np.nan

    for i, row in fat.iterrows():
        base = baseline_for_row(row, fat)
        if pd.notna(fat.at[i, "gct_num"]) and base["gct_num"].notna().sum() >= 15:
            fat.at[i, "fatigue_gct"] = float(robust_z(pd.Series([fat.at[i, "gct_num"]]), base["gct_num"]).iloc[0])
        if pd.notna(fat.at[i, "vr_num"]) and base["vr_num"].notna().sum() >= 15:
            fat.at[i, "fatigue_vr"] = float(robust_z(pd.Series([fat.at[i, "vr_num"]]), base["vr_num"]).iloc[0])
        if pd.notna(fat.at[i, "cad_num"]) and base["cad_num"].notna().sum() >= 15:
            z = float(robust_z(pd.Series([fat.at[i, "cad_num"]]), base["cad_num"]).iloc[0])
            fat.at[i, "fatigue_cad"] = abs(z)
        if pd.notna(fat.at[i, "hr_per_pace"]) and base["hr_per_pace"].notna().sum() >= 20:
            fat.at[i, "fatigue_hr"] = float(robust_z(pd.Series([fat.at[i, "hr_per_pace"]]), base["hr_per_pace"]).iloc[0])

    for c in ["fatigue_gct", "fatigue_vr", "fatigue_cad", "fatigue_hr"]:
        fat[c] = fat[c].fillna(0.0)

    raw_f = (
        0.35 * fat["fatigue_gct"] +
        0.30 * fat["fatigue_vr"] +
        0.20 * fat["fatigue_cad"] +
        0.15 * fat["fatigue_hr"]
    )
    p5, p95 = np.nanpercentile(raw_f, 5), np.nanpercentile(raw_f, 95)
    fat["Fatigue_score"] = (100 * (raw_f - p5) / (p95 - p5 + 1e-9)).clip(0, 100)
    fat["Fatigue_flag"] = fat["Fatigue_score"] > 65

    def fatigue_type(row):
        mech = row.get("fatigue_gct", 0) + row.get("fatigue_vr", 0)
        cardio = row.get("fatigue_hr", 0)
        if mech > 1.5 and cardio < 0.5:
            return "mechanical"
        if cardio > 1.2 and mech < 1.0:
            return "cardio"
        if mech > 1.2 and cardio > 1.0:
            return "mixed"
        return "none"

    fat["Fatigue_type"] = fat.apply(fatigue_type, axis=1)

    df.loc[fat.index, "Fatigue_score"] = fat["Fatigue_score"].values
    df.loc[fat.index, "Fatigue_flag"] = fat["Fatigue_flag"].values
    df.loc[fat.index, "Fatigue_type"] = fat["Fatigue_type"].values

# =========================================================
# 8) DASHBOARD (UX: kevesebb táblázat, több vizuál)
# =========================================================
st.title("🏃 Garmin Futás Dashboard")

# -------------------------
# Segédek (biztos, hogy vannak)
# -------------------------
def num(v):
    return pd.to_numeric(
        pd.Series([v]).astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    ).iloc[0]

def med(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    ).median()

# -------------------------
# TABOK: Mobil / Desktop
# -------------------------
# Mobilon tabok, desktopon is ugyanaz (egységes UX)
tab_overview, tab_last, tab_warn, tab_ready, tab_data = st.tabs(
    ["📌 Áttekintés", "🔎 Utolsó futás", "🚦 Warning", "🏁 Readiness", "📄 Adatok"]
)

# -------------------------
# Minimális adat: csak érvényes dátum kell a nézethez
# -------------------------
if "Dátum" not in df.columns:
    st.error("Nem találom a 'Dátum' oszlopot.")
    st.stop()

d = df[df["Dátum"].notna()].copy()
if d.empty:
    st.error("Nincs érvényes dátummal rendelkező sor (Dátum parse -> NaT).")
    st.stop()

run_type_col = safe_col(d, "Run_type")
fatigue_col = safe_col(d, "Fatigue_score")
fatigue_type_col = safe_col(d, "Fatigue_type")
slope_col = safe_col(d, "slope_bucket")

# -------------------------
# Sidebar: Szűrők + Baseline
# -------------------------
st.sidebar.divider()
# -------------------------
# HRmax (globális, hogy minden tabban létezzen)
# -------------------------
if "hrmax" not in st.session_state:
    st.session_state.hrmax = 190  # default

st.sidebar.divider()
st.sidebar.header("Intenzitás (HR zónák)")

st.session_state.hrmax = st.sidebar.number_input(
    "HRmax (ütés/perc) – zónákhoz",
    min_value=120,
    max_value=240,
    value=int(st.session_state.hrmax),
    step=1,
    help="Ha nem tudod pontosan, hagyd 185–195 körül; a zónák becslések."
)

hrmax = int(st.session_state.hrmax)  # <- EZT használja majd a heti intenzitás blokk

st.sidebar.header("Szűrők")

min_date = pd.to_datetime(d["Dátum"], errors="coerce").dropna().min()
max_date = pd.to_datetime(d["Dátum"], errors="coerce").dropna().max()

if pd.isna(min_date) or pd.isna(max_date):
    st.sidebar.error("Nem sikerült dátumtartományt képezni (NaT).")
    st.stop()

date_from, date_to = st.sidebar.date_input(
    "Dátumtartomány",
    value=(min_date.date(), max_date.date()),
)

mask = (d["Dátum"].dt.date >= date_from) & (d["Dátum"].dt.date <= date_to)

if run_type_col:
    types = ["easy", "tempo", "race"]
    present = [t for t in types if t in set(d[run_type_col].dropna().astype(str))]
    selected_types = st.sidebar.multiselect("Run_type", options=present, default=present)
    if selected_types:
        mask &= d[run_type_col].isin(selected_types)

if slope_col:
    slope_opts = [x for x in d[slope_col].dropna().unique().tolist()]
    slope_sel = st.sidebar.multiselect("Terep (slope_bucket)", options=slope_opts, default=slope_opts)
    if slope_sel:
        mask &= d[slope_col].isin(slope_sel)

if fatigue_col and d[fatigue_col].notna().sum() > 0:
    fmin = float(np.nanmin(d[fatigue_col]))
    fmax = float(np.nanmax(d[fatigue_col]))
    sel_fmin, sel_fmax = st.sidebar.slider(
        "Fatigue_score", min_value=fmin, max_value=fmax, value=(fmin, fmax)
    )
    mask &= d[fatigue_col].between(sel_fmin, sel_fmax)

st.sidebar.divider()
st.sidebar.header("Baseline (B-line)")

baseline_weeks = st.sidebar.slider(
    "Baseline ablak (hetek)", min_value=4, max_value=52, value=12, step=1,
    help="Ennyi hét easy futásai alapján számoljuk a baseline mediánt."
)
baseline_min_runs = st.sidebar.slider(
    "Minimum baseline futás", min_value=10, max_value=80, value=25, step=5,
    help="Ha az ablakban nincs elég easy futás, fallback: legutóbbi N easy futás."
)

view = d.loc[mask].copy().sort_values("Dátum")

# -------------------------
# ÁTTEKINTÉS: KPI + grafikonok
# -------------------------
with tab_overview:
    st.subheader("🗓️ Napi összkép (coach)")

    # daily_coach_summary opcionális – ha nincs definiálva, ne dőljön össze
    if "daily_coach_summary" in globals():
        status, msg = daily_coach_summary(
            base_all=d,
            run_type_col=run_type_col,
            fatigue_col=fatigue_col,
            baseline_weeks=baseline_weeks,
            baseline_min_runs=baseline_min_runs
        )
        if status == "🔴":
            st.error(f"{status} {msg}")
        elif status == "🟠":
            st.warning(f"{status} {msg}")
        else:
            st.success(f"{status} {msg}")
    else:
        st.info("ℹ️ (Opció) daily_coach_summary nincs bekötve – csak a grafikonok/KPI futnak.")

    st.divider()

    # =========================================================
    # 📊 Heti terhelés & ramp rate
    # =========================================================
    st.subheader("📊 Heti terhelés & ramp rate")

    base_all = d.copy()
    base_all = base_all[base_all["Dátum"].notna()].sort_values("Dátum")

    # --- távolság km
    if "dist_km" not in base_all.columns:
        if "Távolság" in base_all.columns:
            base_all["dist_km"] = to_float_series(base_all["Távolság"])
        else:
            base_all["dist_km"] = np.nan

    # --- emelkedés m
    if "asc_m" not in base_all.columns:
        if "Teljes emelkedés" in base_all.columns:
            base_all["asc_m"] = to_float_series(base_all["Teljes emelkedés"])
        else:
            base_all["asc_m"] = np.nan

    # --- idő sec (Idő / Menetidő / Eltelt idő)
    time_col_candidates = [c for c in ["Idő", "Menetidő", "Eltelt idő"] if c in base_all.columns]
    if time_col_candidates:
        time_col = time_col_candidates[0]
        base_all["dur_sec"] = base_all[time_col].apply(duration_to_seconds)
    else:
        base_all["dur_sec"] = np.nan

    # --- Heti aggregálás (hétfői hétkezdéssel)
    w = base_all.set_index("Dátum")
    weekly = (
        w.resample("W-MON")
         .agg(week_km=("dist_km", "sum"),
              week_sec=("dur_sec", "sum"),
              week_elev=("asc_m", "sum"))
         .reset_index()
         .rename(columns={"Dátum": "week"})
         .sort_values("week")
    )

    weekly["week_hours"] = weekly["week_sec"] / 3600.0

    # --- Ramp rate: utolsó (teljes) hét vs előző 4 hét átlaga
    rr_km = rr_time = rr_elev = np.nan

    if len(weekly) >= 6:
        last_idx = len(weekly) - 1

        # ha a legutolsó hét "félkész" (nagyon kicsi), lépjünk vissza egyet
        if weekly.loc[last_idx, "week_km"] < 0.3 * max(1e-9, weekly.loc[last_idx-1, "week_km"]):
            last_idx -= 1

        last_week = weekly.loc[last_idx]
        prev4 = weekly.loc[last_idx-4:last_idx-1]

        def ramp(curr, prev_mean):
            if pd.isna(curr) or pd.isna(prev_mean) or prev_mean <= 0:
                return np.nan
            return (curr - prev_mean) / prev_mean * 100.0

        rr_km = ramp(float(last_week["week_km"]), float(np.nanmean(prev4["week_km"])))
        rr_time = ramp(float(last_week["week_hours"]), float(np.nanmean(prev4["week_hours"])))
        rr_elev = ramp(float(last_week["week_elev"]), float(np.nanmean(prev4["week_elev"])))

    # --- UI: metrika választó
    metric = st.selectbox(
        "Melyik terhelést nézzük?",
        ["Heti km", "Heti idő (óra)", "Heti emelkedés (m)"],
        index=0,
        key="load_metric"
    )

    if metric == "Heti km":
        y = "week_km"
        rr = rr_km
        ylabel = "km"
    elif metric == "Heti idő (óra)":
        y = "week_hours"
        rr = rr_time
        ylabel = "óra"
    else:
        y = "week_elev"
        rr = rr_elev
        ylabel = "m"

    # --- Ramp badge
    def ramp_badge(rr):
        if pd.isna(rr):
            return ("⚪", "Ramp rate: nincs elég adat (legalább 6 hét kell)")
        if rr <= 8:
            return ("🟢", f"Ramp rate: {rr:+.1f}% (biztonságos)")
        if rr <= 12:
            return ("🟠", f"Ramp rate: {rr:+.1f}% (figyelmeztető)")
        return ("🔴", f"Ramp rate: {rr:+.1f}% (túl gyors emelés)")

    badge, badge_txt = ramp_badge(rr)
    st.caption(f"{badge} {badge_txt}")

    # --- Heti oszlopdiagram
    fig_week = px.bar(
        weekly,
        x="week",
        y=y,
        title=f"Heti terhelés – {metric}",
        labels={"week": "Hét", y: ylabel},
    )
    st.plotly_chart(fig_week, use_container_width=True)

    # --- Napi gördülő 7/28 napos trend (km / idő / elev)
    daily = base_all.copy()
    daily["date"] = daily["Dátum"].dt.date
    daily = (
        daily.groupby("date", as_index=False)
             .agg(km=("dist_km", "sum"),
                  sec=("dur_sec", "sum"),
                  elev=("asc_m", "sum"))
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    daily["km_7"] = daily["km"].rolling(7, min_periods=3).sum()
    daily["km_28"] = daily["km"].rolling(28, min_periods=10).sum()

    daily["h_7"] = (daily["sec"].rolling(7, min_periods=3).sum()) / 3600.0
    daily["h_28"] = (daily["sec"].rolling(28, min_periods=10).sum()) / 3600.0

    daily["elev_7"] = daily["elev"].rolling(7, min_periods=3).sum()
    daily["elev_28"] = daily["elev"].rolling(28, min_periods=10).sum()

    if metric == "Heti km":
        fig_roll = px.line(daily, x="date", y=["km_7", "km_28"],
                           title="Gördülő összeg – 7 nap vs 28 nap (km)")
    elif metric == "Heti idő (óra)":
        fig_roll = px.line(daily, x="date", y=["h_7", "h_28"],
                           title="Gördülő összeg – 7 nap vs 28 nap (óra)")
    else:
        fig_roll = px.line(daily, x="date", y=["elev_7", "elev_28"],
                           title="Gördülő összeg – 7 nap vs 28 nap (emelkedés)")

    st.plotly_chart(fig_roll, use_container_width=True)

    with st.expander("📋 Heti táblázat (részletek)"):
        st.dataframe(weekly.tail(24), use_container_width=True, hide_index=True)
    # =========================================================
    # ⚡ Intenzitás megoszlás heti szinten
    # easy/tempo/race arány + HR zónák
    # =========================================================
    st.subheader("⚡ Intenzitás megoszlás heti szinten")

    base_all = d.copy()
    base_all = base_all[base_all["Dátum"].notna()].sort_values("Dátum")

    # --- futás szűrés (ha van Tevékenység típusa)
    if "Tevékenység típusa" in base_all.columns:
        m_run = base_all["Tevékenység típusa"].astype(str).str.contains("Fut", na=False)
        base_all = base_all[m_run].copy()

    if base_all.empty:
        st.info("Nincs futás adat az intenzitás bontáshoz.")
    else:
        # --- távolság (km) ha kell
        if "dist_km" not in base_all.columns:
            if "Távolság" in base_all.columns:
                base_all["dist_km"] = to_float_series(base_all["Távolság"])
            else:
                base_all["dist_km"] = np.nan

        # --- HR (átlag)
        hr_col_candidates = [c for c in ["hr_num", "Átlagos pulzusszám", "Átlagos pulzusszám "] if c in base_all.columns]
        if "hr_num" in base_all.columns:
            base_all["hr_avg"] = base_all["hr_num"]
        elif hr_col_candidates:
            base_all["hr_avg"] = to_float_series(base_all[hr_col_candidates[0]])
        else:
            base_all["hr_avg"] = np.nan

        # --- Run_type fallback, ha nincs
        if run_type_col and run_type_col in base_all.columns:
            base_all["rt"] = base_all[run_type_col].astype(str)
        else:
            base_all["rt"] = "unknown"

        # --- HR% és zóna
        base_all["hr_pct"] = np.where(
            base_all["hr_avg"].notna() & (hrmax > 0),
            base_all["hr_avg"] / float(hrmax),
            np.nan
        )
        base_all["hr_zone"] = base_all["hr_pct"].apply(hr_zone_from_pct)

        # --- Heti bontás
        w = base_all.set_index("Dátum")

        # 1) heti összkilométer
        weekly_km = w.resample("W-MON")["dist_km"].sum().rename("week_km").reset_index()

        # 2) heti run_type arány (darabszám alapú)
        rt_week = (
            w.resample("W-MON")["rt"]
             .value_counts()
             .rename("count")
             .reset_index()
             .rename(columns={"Dátum": "week"})
        )
        rt_pivot = rt_week.pivot_table(index="week", columns="rt", values="count", fill_value=0).reset_index()
        rt_cols = [c for c in rt_pivot.columns if c != "week"]
        rt_pivot["total"] = rt_pivot[rt_cols].sum(axis=1)
        for c in rt_cols:
            rt_pivot[c] = np.where(rt_pivot["total"] > 0, rt_pivot[c] / rt_pivot["total"] * 100.0, 0.0)

        # 3) heti HR zóna megoszlás (darabszám alapú)
        hz_week = (
            w.resample("W-MON")["hr_zone"]
             .value_counts()
             .rename("count")
             .reset_index()
             .rename(columns={"Dátum": "week"})
        )
        hz_pivot = hz_week.pivot_table(index="week", columns="hr_zone", values="count", fill_value=0).reset_index()
        hz_cols = [c for c in ["Z1", "Z2", "Z3", "Z4", "Z5"] if c in hz_pivot.columns]
        hz_pivot["total"] = hz_pivot[hz_cols].sum(axis=1) if hz_cols else 0
        for c in hz_cols:
            hz_pivot[c] = np.where(hz_pivot["total"] > 0, hz_pivot[c] / hz_pivot["total"] * 100.0, 0.0)

        # --- UI választók
        cL, cR = st.columns([1, 1])
        with cL:
            st.markdown("#### 🧩 Easy / Tempo / Race arány (heti %)")
            if run_type_col and (("easy" in rt_cols) or ("tempo" in rt_cols) or ("race" in rt_cols)):
                show_rts = [c for c in ["easy", "tempo", "race"] if c in rt_cols]
                rt_long = rt_pivot[["week"] + show_rts].melt("week", var_name="Run_type", value_name="Percent")
                fig_rt = px.bar(
                    rt_long,
                    x="week",
                    y="Percent",
                    color="Run_type",
                    barmode="stack",
                    labels={"week": "Hét", "Percent": "%"},
                )
                st.plotly_chart(fig_rt, use_container_width=True)
            else:
                st.info("Run_type hiányzik / nincs felismerve (easy/tempo/race).")

        with cR:
            st.markdown("#### ❤️ HR zóna megoszlás (heti %)")
            if base_all["hr_zone"].notna().sum() >= 5 and hz_cols:
                hz_long = hz_pivot[["week"] + hz_cols].melt("week", var_name="Zone", value_name="Percent")
                fig_hz = px.bar(
                    hz_long,
                    x="week",
                    y="Percent",
                    color="Zone",
                    barmode="stack",
                    labels={"week": "Hét", "Percent": "%"},
                )
                st.plotly_chart(fig_hz, use_container_width=True)
            else:
                st.info("Kevés / hiányzó pulzus adat a zónákhoz (vagy HRmax nincs megadva).")

        st.divider()

        # --- Kombinált “miért magas a fatigue?” nézet:
        # heti km + magas intenzitás arány (Z4+Z5)
        st.markdown("#### 🔎 Terhelés vs Intenzitás (heti km + Z4/Z5 arány)")

        if hz_cols:
            hz_pivot["hi_intensity_pct"] = 0.0
            if "Z4" in hz_pivot.columns:
                hz_pivot["hi_intensity_pct"] += hz_pivot["Z4"]
            if "Z5" in hz_pivot.columns:
                hz_pivot["hi_intensity_pct"] += hz_pivot["Z5"]

            combo = weekly_km.rename(columns={"Dátum": "week"}).merge(
                hz_pivot[["week", "hi_intensity_pct"]], on="week", how="left"
            )
            combo["hi_intensity_pct"] = combo["hi_intensity_pct"].fillna(0.0)

            fig_combo = px.scatter(
                combo,
                x="week_km",
                y="hi_intensity_pct",
                hover_data=["week"],
                labels={"week_km": "Heti km", "hi_intensity_pct": "Z4+Z5 %"},
            )
            st.plotly_chart(fig_combo, use_container_width=True)

            # gyors “coach” értelmezés
            if len(combo.dropna(subset=["week_km"])) >= 4:
                last = combo.iloc[-1]
                msg = f"Utolsó hét: **{last['week_km']:.1f} km**, magas intenzitás (Z4+Z5): **{last['hi_intensity_pct']:.0f}%**."
                if last["hi_intensity_pct"] >= 30:
                    st.warning("🟠 Sok a magas intenzitás (Z4+Z5) → ez önmagában felnyomhatja a Fatigue_score-t.")
                    st.caption(msg)
                else:
                    st.success("🟢 A magas intenzitás arány nem extrém → ha magas a Fatigue_score, inkább a terhelés/ramp vagy technikai faktor lehet.")
                    st.caption(msg)

        else:
            st.info("HR zóna adatok nélkül a 'Terhelés vs Intenzitás' ábra nem készíthető el.")


    # KPI-k (ha nincs technika/fatigue, akkor is menjen)
    tech_avg = view["Technika_index"].mean() if ("Technika_index" in view.columns and view["Technika_index"].notna().any()) else np.nan
    fat_avg = view[fatigue_col].mean() if (fatigue_col and view[fatigue_col].notna().any()) else np.nan
    most_type = view[run_type_col].value_counts().index[0] if (run_type_col and len(view) > 0) else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Futások (szűrve)", f"{len(view)}")
    c2.metric("Átlag Technika_index", f"{tech_avg:.1f}" if pd.notna(tech_avg) else "—")
    c3.metric("Átlag Fatigue_score", f"{fat_avg:.1f}" if pd.notna(fat_avg) else "—")
    c4.metric("Leggyakoribb típus", most_type)

    st.divider()

    # Technika idősor csak ha van Technika_index
    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("📈 Technika_index időben")
        if "Technika_index" in view.columns and view["Technika_index"].notna().sum() >= 3:
            fig = px.scatter(
                view.dropna(subset=["Technika_index"]),
                x="Dátum",
                y="Technika_index",
                color=run_type_col if run_type_col else None,
                symbol=slope_col if slope_col else None,
                hover_data=[c for c in ["Cím", "Átlagos tempó", fatigue_col, fatigue_type_col, slope_col] if c and c in view.columns],
                opacity=0.75,
            )
            view2 = view[["Dátum", "Technika_index"]].dropna().sort_values("Dátum").copy()
            if len(view2) >= 10:
                view2["roll30"] = view2["Technika_index"].rolling(window=30, min_periods=10).mean()
                fig_line = px.line(view2, x="Dátum", y="roll30")
                for tr in fig_line.data:
                    fig.add_trace(tr)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nincs elég Technika_index adat az idősorhoz.")

    with right:
        st.subheader("🗺️ Terep megoszlás (szűrve)")
        if slope_col and len(view[slope_col].dropna()) > 0:
            cnt = view[slope_col].value_counts().reset_index()
            cnt.columns = ["slope_bucket", "db"]
            st.plotly_chart(px.bar(cnt, x="slope_bucket", y="db"), use_container_width=True)
        else:
            st.info("Nincs slope_bucket adat.")

    st.divider()

    cA, cB = st.columns(2)
    with cA:
        st.subheader("🧭 Technika vs Fáradás (kvadráns)")
        if ("Technika_index" in view.columns and fatigue_col and
            view["Technika_index"].notna().sum() >= 10 and view[fatigue_col].notna().sum() >= 10):
            dd = view.dropna(subset=["Technika_index", fatigue_col]).copy()
            tech_med = float(np.nanmedian(dd["Technika_index"]))
            fat_med = float(np.nanmedian(dd[fatigue_col]))

            fig2 = px.scatter(
                dd,
                x="Technika_index",
                y=fatigue_col,
                color=run_type_col if run_type_col else None,
                symbol=slope_col if slope_col else None,
                hover_data=[c for c in ["Dátum", "Cím", "Átlagos tempó", fatigue_type_col, slope_col] if c and c in dd.columns],
                opacity=0.75
            )
            fig2.add_vline(x=tech_med)
            fig2.add_hline(y=fat_med)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"Medián határok: Technika {tech_med:.1f}, Fatigue {fat_med:.1f}")
        else:
            st.info("Kevés Technika/Fatigue adat a kvadránshoz.")
    # =========================================================
# TERHELÉS vs TECHNIKA (heti szinten)
# =========================================================
st.divider()
st.subheader("🧭 Terhelés vs Technika (heti trend)")

# szükséges oszlopok
need_cols = ["Dátum", "Technika_index"]
if not all(c in d.columns for c in need_cols):
    st.info("Nincs elég adat a terhelés–technika elemzéshez.")
else:
    # ---- heti aggregálás
    w = d.dropna(subset=["Dátum", "Technika_index"]).copy()
    w["week"] = w["Dátum"].dt.to_period("W").astype(str)

    agg = {
        "Technika_index": "mean"
    }

    # terhelés forrás kiválasztása
    load_col = None
    if "Távolság" in w.columns:
        load_col = "Távolság"
        agg[load_col] = "sum"
        load_label = "Heti táv (km)"
    elif "Idő" in w.columns:
        load_col = "Idő"
        agg[load_col] = "sum"
        load_label = "Heti idő"
    else:
        st.info("Nincs Távolság vagy Idő oszlop a terheléshez.")
        load_col = None

    if load_col:
        weekly = w.groupby("week").agg(agg).reset_index()
        weekly = weekly.sort_values("week")

        if len(weekly) < 6:
            st.info("Kevés heti adat (min. ~6 hét ajánlott).")
        else:
            # ---- trend számítás (utolsó 6 hét)
            last = weekly.tail(6)

            tech_trend = np.polyfit(range(len(last)), last["Technika_index"], 1)[0]
            load_trend = np.polyfit(range(len(last)), last[load_col], 1)[0]

            # ---- értelmezés
            if load_trend > 0 and tech_trend < 0:
                verdict = "🔴 Terhelés nő, technika romlik → túlterhelés gyanú"
            elif load_trend > 0 and tech_trend < 0.05:
                verdict = "🟠 Terhelés nő, technika stagnál → határon"
            elif load_trend > 0 and tech_trend > 0:
                verdict = "🟢 Terhelés nő, technika javul → adaptáció"
            else:
                verdict = "ℹ️ Nincs egyértelmű trend"

            st.markdown(f"### {verdict}")

            # ---- vizualizáció
            fig = px.scatter(
                weekly,
                x=load_col,
                y="Technika_index",
                trendline="ols",
                hover_data=["week"],
                labels={
                    load_col: load_label,
                    "Technika_index": "Heti átlag Technika_index"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- kis magyarázat
            st.caption(
                "🔍 Minden pont egy hét. "
                "Balról jobbra nő a terhelés, fentről lefelé romlik a technika."
            )


    with cB:
        st.subheader("🏅 Top / Bottom futások")
        with st.expander("Megnyitás", expanded=False):
            topn = st.slider("N", 5, 30, 10, key="topn_overview")
            if "Technika_index" in view.columns and view["Technika_index"].notna().any():
                cols = ["Dátum"]
                if run_type_col: cols.append(run_type_col)
                if slope_col: cols.append(slope_col)
                cols += ["Technika_index"]
                if fatigue_col: cols.append(fatigue_col)
                if "Cím" in view.columns: cols.append("Cím")

                top = view.sort_values("Technika_index", ascending=False).head(topn)
                bot = view.sort_values("Technika_index", ascending=True).head(topn)

                st.markdown("**⬆️ Top technika**")
                st.dataframe(top[cols], use_container_width=True, hide_index=True, height=260)
                st.markdown("**⬇️ Bottom technika**")
                st.dataframe(bot[cols], use_container_width=True, hide_index=True, height=260)
            else:
                st.info("Nincs Technika_index a top/bottom listához.")

# -------------------------
# UTOLSÓ FUTÁS: vizuális baseline összevetés + jelzések
# -------------------------
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
                title = r["Cím"] if "Cím" in options.columns and pd.notna(r.get("Cím")) else ""
                rt = r["Run_type"] if "Run_type" in options.columns and pd.notna(r.get("Run_type")) else ""
                return f"{r['Dátum'].strftime('%Y-%m-%d %H:%M')} | {rt} | {title}"[:120]

            options["__label"] = options.apply(label_row, axis=1)
            chosen_label = st.selectbox("Futás kiválasztása", options["__label"].tolist(), index=len(options)-1, key="pick_last")
            last = options.loc[options["__label"] == chosen_label].iloc[0]

            baseline_full = get_easy_baseline(
                base_df=base,
                last_date=last["Dátum"],
                weeks=baseline_weeks,
                min_runs=baseline_min_runs
            )

            # KPI-k
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Technika_index", f"{float(last['Technika_index']):.1f}")
            c2.metric("Fatigue_score", f"{float(last.get('Fatigue_score')):.1f}" if pd.notna(last.get("Fatigue_score")) else "—")
            c3.metric("Run_type", str(last.get("Run_type")) if pd.notna(last.get("Run_type")) else "—")
            pace = last.get("Átlagos tempó") if "Átlagos tempó" in base.columns else None
            dist = last.get("Távolság") if "Távolság" in base.columns else None
            c4.metric("Tempó / Táv", f"{pace} / {dist} km" if (pd.notna(pace) or pd.notna(dist)) else "—")

            if "Cím" in base.columns and pd.notna(last.get("Cím")):
                st.caption(f"**Cím:** {last['Cím']}")

            st.caption(f"Baseline easy futások száma: {len(baseline_full)} (hetek: {baseline_weeks})")

            if len(baseline_full) < 8:
                st.info("Kevés baseline easy futás – az összevetés bizonytalan. (Ajánlott ≥ 8–10)")
            else:
                compare_cols = [
                    ("Átl. pedálütem", "Cadence (spm)", "cadence_stability"),
                    ("Átlagos lépéshossz", "Lépéshossz (m)", "higher_better"),
                    ("Átlagos függőleges arány", "Vertical Ratio (%)", "lower_better"),
                    ("Átlagos függőleges oszcilláció", "Vertical Osc (cm)", "lower_better"),
                    ("Átlagos talajérintési idő", "GCT (ms)", "lower_better"),
                    ("Átlagos pulzusszám", "Átlag pulzus", "context"),
                    ("Max. pulzus", "Max pulzus", "context"),
                ]

                rows = []
                for col, label, rule in compare_cols:
                    if col not in base.columns:
                        continue
                    v = num(last.get(col))
                    b = med(baseline_full[col])
                    if pd.isna(v) or pd.isna(b) or b == 0:
                        continue

                    delta_pct = (v - b) / b * 100.0

                    if rule == "lower_better":
                        good = -delta_pct
                    elif rule == "higher_better":
                        good = delta_pct
                    elif rule == "cadence_stability":
                        good = -abs(delta_pct)
                    else:
                        good = 0.0

                    rows.append([label, float(v), float(b), float(delta_pct), float(good), rule])

                if rows:
                    comp = pd.DataFrame(rows, columns=["Mutató", "Utolsó", "Baseline", "Eltérés_%", "Jó_irány", "rule"])

                    st.markdown("### 📈 Eltérések az easy baseline-hoz képest")
                    figd = px.bar(
                        comp.sort_values("Jó_irány"),
                        x="Jó_irány",
                        y="Mutató",
                        orientation="h",
                        hover_data=["Utolsó", "Baseline", "Eltérés_%", "rule"]
                    )
                    st.plotly_chart(figd, use_container_width=True)

                    st.markdown("### 🚩 Gyors jelzések")
                    msgs = []
                    for _, r in comp.iterrows():
                        if r["rule"] in ("lower_better", "higher_better"):
                            if r["Jó_irány"] < -5:
                                msgs.append(f"🔴 **{r['Mutató']}** romlott (≈ {r['Eltérés_%']:+.1f}%).")
                            elif r["Jó_irány"] < -2:
                                msgs.append(f"🟠 **{r['Mutató']}** kicsit romlott (≈ {r['Eltérés_%']:+.1f}%).")
                            else:
                                msgs.append(f"🟢 **{r['Mutató']}** rendben (≈ {r['Eltérés_%']:+.1f}%).")
                        elif r["rule"] == "cadence_stability":
                            if abs(r["Eltérés_%"]) > 5:
                                msgs.append(f"🟠 **{r['Mutató']}** eltér a baseline-tól (≈ {r['Eltérés_%']:+.1f}%).")
                            else:
                                msgs.append(f"🟢 **{r['Mutató']}** stabil (≈ {r['Eltérés_%']:+.1f}%).")

                    for m in msgs[:10]:
                        st.write(m)

                    with st.expander("📋 Részletes táblázat"):
                        st.dataframe(comp.drop(columns=["Jó_irány"]), use_container_width=True, hide_index=True)
                else:
                    st.info("Nincs elég összehasonlítható metrika az utolsó futáshoz.")

# -------------------------
# WARNING TAB
# -------------------------
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

        base_all = d.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
        if run_type_col:
            easy = base_all[base_all[run_type_col] == "easy"].copy()
        else:
            easy = base_all.copy()

        # baseline-window szűkítés (ne a teljes múlt)
        if len(easy) > 0:
            last_day = easy["Dátum"].max()
            start = last_day - pd.Timedelta(weeks=baseline_weeks)
            easy = easy[easy["Dátum"] >= start].copy()
            if len(easy) < baseline_min_runs:
                easy = easy.tail(baseline_min_runs).copy()

        easy_f = easy.dropna(subset=[fatigue_col]).copy()
        if len(easy_f) < 5:
            st.info("Nincs elég easy + Fatigue_score adat (legalább ~5 futás).")
        else:
            last_red = easy_f.tail(n_red).copy()
            last_yellow = easy_f.tail(n_yellow).copy()

            last_red["hit_red"] = (last_red["Technika_index"] < tech_red) & (last_red[fatigue_col] > fat_red)
            last_yellow["hit_yellow"] = (last_yellow["Technika_index"] < tech_yellow) | (last_yellow[fatigue_col] > fat_yellow)

            red_hits = int(last_red["hit_red"].sum())
            yellow_hits = int(last_yellow["hit_yellow"].sum())

            status = "🟢 ZÖLD"
            reason = "Stabil easy technika / fáradás kontrollált."
            if red_hits >= need_red:
                status = "🔴 PIROS"
                reason = f"Utolsó {n_red} easy futásból {red_hits} találat: Tech < {tech_red} ÉS Fatigue > {fat_red}."
            elif yellow_hits >= need_yellow:
                status = "🟠 SÁRGA"
                reason = f"Utolsó {n_yellow} easy futásból {yellow_hits} találat: Tech < {tech_yellow} VAGY Fatigue > {fat_yellow}."

            if status.startswith("🔴"):
                st.error(f"{status} — {reason}")
            elif status.startswith("🟠"):
                st.warning(f"{status} — {reason}")
            else:
                st.success(f"{status} — {reason}")

            c1, c2 = st.columns(2)
            c1.metric("PIROS találatok", f"{red_hits}/{n_red}")
            c2.metric("SÁRGA találatok", f"{yellow_hits}/{n_yellow}")

            show = easy_f.tail(max(n_yellow, n_red)).copy()
            show["red_hit"] = (show["Technika_index"] < tech_red) & (show[fatigue_col] > fat_red)
            show["yellow_hit"] = (show["Technika_index"] < tech_yellow) | (show[fatigue_col] > fat_yellow)

            figw = px.scatter(
                show,
                x="Dátum",
                y="Technika_index",
                color="yellow_hit",
                symbol="red_hit",
                hover_data=["Cím"] if "Cím" in show.columns else None
            )
            st.plotly_chart(figw, use_container_width=True)

            with st.expander("📋 Részletek"):
                cols = ["Dátum", "Technika_index", fatigue_col, "red_hit", "yellow_hit"]
                if "Cím" in show.columns:
                    cols.append("Cím")
                st.dataframe(show.sort_values("Dátum", ascending=False)[cols], use_container_width=True, hide_index=True)

# -------------------------
# READINESS TAB
# -------------------------
with tab_ready:
    st.subheader("🏁 Verseny-előrejelzés (Readiness) – 14 napos ablak")

    if run_type_col is None or fatigue_col is None or "Technika_index" not in d.columns:
        st.info("Readiness-hez kell Run_type + Fatigue_score + Technika_index.")
    else:
        base_all = d.dropna(subset=["Dátum", "Technika_index"]).sort_values("Dátum")
        easy = base_all[base_all[run_type_col] == "easy"].dropna(subset=[fatigue_col]).copy()

        if len(easy) < 10:
            st.info("Nincs elég easy + Fatigue_score adat a readiness-hez.")
        else:
            default_date = base_all["Dátum"].max().date()
            race_date = st.date_input("Verseny dátuma", value=default_date, key="race_date")
            window_days = st.slider("Ablak (nap)", 7, 28, 14, key="race_window")

            start = pd.Timestamp(race_date) - pd.Timedelta(days=window_days)
            end = pd.Timestamp(race_date)
            w = easy[(easy["Dátum"] >= start) & (easy["Dátum"] <= end)].copy()

            if len(w) < 3:
                st.warning(f"Az utolsó {window_days} napban kevés easy futás van ({len(w)}).")
            else:
                tech_mean = float(w["Technika_index"].mean())
                fat_mean = float(w[fatigue_col].mean())

                w2 = w.sort_values("Dátum")[["Dátum", "Technika_index"]].dropna().copy()
                x = (w2["Dátum"] - w2["Dátum"].min()).dt.total_seconds().to_numpy()
                y = w2["Technika_index"].to_numpy()
                slope_per_day = 0.0
                if len(w2) >= 3 and np.nanstd(x) > 0:
                    slope = float(np.polyfit(x, y, 1)[0])
                    slope_per_day = slope * 86400.0

                red_hits = int(((w["Technika_index"] < 35) & (w[fatigue_col] > 55)).sum())

                tech_p25, tech_p75 = np.nanpercentile(easy["Technika_index"], [25, 75])
                fat_p25, fat_p75 = np.nanpercentile(easy[fatigue_col], [25, 75])

                def scale_up(v, lo, hi):
                    return float(np.clip(100 * (v - lo) / (hi - lo + 1e-9), 0, 100))

                def scale_down(v, lo, hi):
                    return float(np.clip(100 * (hi - v) / (hi - lo + 1e-9), 0, 100))

                tech_score = scale_up(tech_mean, tech_p25, tech_p75)
                fat_score = scale_down(fat_mean, fat_p25, fat_p75)
                trend_score = float(np.clip(50 + 20 * slope_per_day, 0, 100))
                penalty = min(30, red_hits * 10)

                readiness = 0.50 * tech_score + 0.30 * fat_score + 0.20 * trend_score - penalty
                readiness = float(np.clip(readiness, 0, 100))

                status = "🟢 ZÖLD"
                if readiness < 40 or red_hits >= 2:
                    status = "🔴 PIROS"
                elif readiness < 60 or red_hits >= 1:
                    status = "🟠 SÁRGA"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Readiness_score", f"{readiness:.0f}")
                c2.metric("Tech (easy) átlag", f"{tech_mean:.1f}")
                c3.metric("Fatigue (easy) átlag", f"{fat_mean:.1f}")
                c4.metric("Tech trend (pont/nap)", f"{slope_per_day:+.2f}")

                if status.startswith("🟢"):
                    st.success("🟢 Jó verseny-készenlét (taper + stabil easy).")
                elif status.startswith("🟠"):
                    st.warning("🟠 Közepes készenlét (még van fáradás / instabil technika).")
                else:
                    st.error("🔴 Nem ideális (fáradás magas és/vagy technika szétesik easy-n is).")

                fig_r = px.line(w.sort_values("Dátum"), x="Dátum", y=["Technika_index", fatigue_col])
                st.plotly_chart(fig_r, use_container_width=True)

                with st.expander("📋 Ablakban lévő futások"):
                    show_cols = ["Dátum", "Technika_index", fatigue_col]
                    if "Cím" in w.columns:
                        show_cols.append("Cím")
                    st.dataframe(w.sort_values("Dátum", ascending=False)[show_cols], use_container_width=True, hide_index=True)

# -------------------------
# ADATOK TAB (táblázat csak itt)
# -------------------------
with tab_data:
    st.subheader("📄 Adatok (szűrve)")
    st.caption("Itt vannak a részletes táblázatok – az elemzésekhez elég az első 4 tab.")
    st.dataframe(view, use_container_width=True, hide_index=True, height=520)

# -------------------------
# Logout
# -------------------------
st.sidebar.divider()
if st.sidebar.button("Kijelentkezés"):
    st.session_state.auth_ok = False
    st.rerun()
