import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import csv
import urllib.parse
import urllib.request
import json
import time
from datetime import datetime, timezone

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
    "weight_kg_default": 70,
    "height_cm_default": 175,
    "age_default": 40,
    "ramp_warn": 8.0,
    "ramp_red": 12.0,
    "daily_coach_tech_bad": -5.0,
    "daily_coach_tech_warn": -2.0,
    "daily_coach_fat_bad": 60.0,
    "daily_coach_fat_warn": 45.0,
    "daily_coach_trend_bad": -6.0,
    "acwr_acute_days": 7,
    "acwr_chronic_days": 28,
    "acwr_safe_lo": 0.8,
    "acwr_safe_hi": 1.3,
    "acwr_warn_hi": 1.5,
    "ctl_decay": 42,
    "atl_decay": 7,
    "tsb_green": 5,
    "tsb_red": -30,
    "recovery_lookback_weeks": 24,
    "recovery_min_events": 8,
    "asym_warn_pct": 3.0,
    "asym_red_pct": 5.0,
    "optload_lag_weeks": 2,
    "optload_min_weeks": 10,
    "intratio_lag_weeks": 3,
    "easy_target_hr_pct_lo": 0.68,
    "easy_target_hr_pct_hi": 0.76,
    "easy_fatigue_penalty_hr": 3.0,
    "easy_fatigue_penalty_pace": 4.0,
    "easy_fatigue_penalty_pwr": 5.0,
    "easy_target_band_hr": 5,
    "easy_target_band_pace": 7,
    "easy_target_band_pwr": 10,
}

# =========================================================
# STRAVA INTEGRÁCIÓ
# =========================================================
_STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
_STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
_STRAVA_API_BASE = "https://www.strava.com/api/v3"

_STRAVA_TYPE_MAP = {
    "Run": "easy",
    "TrailRun": "easy",
    "VirtualRun": "easy",
    "Race": "race",
    "Workout": "tempo",
}

_STRAVA_FIELD_MAP = {
    "Cím":                    "name",
    "Tevékenység típusa":     "_type",
    "Távolság":               "_dist_m",
    "Idő":                    "_dur_sec_raw",
    "Átlagos tempó":          "_pace_raw",
    "Átlagos pulzusszám":     "average_heartrate",
    "Max pulzus":             "max_heartrate",
    "Teljes emelkedés":       "total_elevation_gain",
    "Átl. pedálütem":         "average_cadence",
    "Átl. teljesítmény":      "average_watts",
    "Max. teljesítmény":      "max_watts",
    "Hőmérséklet":            "average_temp",
    "Dátum":                  "start_date_local",
}


def _strava_secrets() -> tuple[str | None, str | None, str | None]:
    cid   = st.secrets.get("STRAVA_CLIENT_ID", None)
    csec  = st.secrets.get("STRAVA_CLIENT_SECRET", None)
    rtok  = st.secrets.get("STRAVA_REFRESH_TOKEN", None)
    return cid, csec, rtok


def strava_auth_url(client_id: str, redirect_uri: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": "read,activity:read_all",
    }
    return f"{_STRAVA_AUTH_URL}?{urllib.parse.urlencode(params)}"


def strava_exchange_code(client_id: str, client_secret: str, code: str) -> dict | None:
    payload = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
    }).encode()
    req = urllib.request.Request(_STRAVA_TOKEN_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        st.error(f"Strava token csere sikertelen: {e}")
        return None


def strava_refresh_token(client_id: str, client_secret: str, refresh_token: str) -> dict | None:
    payload = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request(_STRAVA_TOKEN_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        st.error(f"Strava token megújítás sikertelen: {e}")
        return None


def _strava_get(endpoint: str, access_token: str, params: dict | None = None) -> list | dict | None:
    url = f"{_STRAVA_API_BASE}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {access_token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            st.warning("Strava API rate limit elérve – várj 15 percet.")
        elif e.code == 401:
            st.error("Strava token érvénytelen – csatlakozz újra.")
        else:
            st.error(f"Strava API hiba {e.code}: {e.reason}")
        return None
    except Exception as e:
        st.error(f"Strava API kapcsolati hiba: {e}")
        return None


@st.cache_data(show_spinner="Strava aktivitások letöltése…", ttl=1800)
def fetch_strava_activities(access_token: str, days_back: int = 365) -> list[dict]:
    after_ts = int(time.time()) - days_back * 86400
    all_acts = []
    page = 1
    while True:
        batch = _strava_get(
            "/athlete/activities",
            access_token,
            params={"after": after_ts, "per_page": 100, "page": page},
        )
        if not batch:
            break
        runs = [a for a in batch if a.get("sport_type", a.get("type", "")) in
                ("Run", "TrailRun", "VirtualRun", "Race", "Workout")]
        all_acts.extend(runs)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.3)
    return all_acts


def strava_activities_to_df(activities: list[dict]) -> pd.DataFrame:
    if not activities:
        return pd.DataFrame()

    rows = []
    for a in activities:
        dist_m = float(a.get("distance", 0) or 0)
        dist_km = dist_m / 1000.0
        dur_sec = float(a.get("moving_time", 0) or 0)
        avg_speed = float(a.get("average_speed", 0) or 0)
        if avg_speed > 0:
            pace_sec_km = 1000.0 / avg_speed
            pace_str = sec_to_pace_str(pace_sec_km)
        else:
            pace_sec_km = np.nan
            pace_str = np.nan

        date_raw = a.get("start_date_local", a.get("start_date", ""))
        try:
            dt = pd.to_datetime(date_raw, utc=False, errors="coerce")
            if dt is not pd.NaT and dt.tzinfo is not None:
                dt = dt.tz_convert("UTC").tz_localize(None)
        except Exception:
            dt = pd.NaT

        cad_raw = a.get("average_cadence")
        cad_num = float(cad_raw) * 2 if cad_raw is not None else np.nan
        hr_num = float(a.get("average_heartrate") or np.nan) if a.get("average_heartrate") else np.nan
        power_avg = float(a.get("average_watts") or np.nan) if a.get("average_watts") else np.nan
        power_max = float(a.get("max_watts") or np.nan) if a.get("max_watts") else np.nan
        asc_m = float(a.get("total_elevation_gain") or 0)
        sport_type = a.get("sport_type", a.get("type", "Run"))
        run_type_raw = _STRAVA_TYPE_MAP.get(sport_type, "easy")

        rows.append({
            "Dátum": dt,
            "Cím": a.get("name", ""),
            "Tevékenység típusa": "Futás",
            "dist_km": dist_km if dist_km > 0 else np.nan,
            "dur_sec": dur_sec if dur_sec > 0 else np.nan,
            "pace_sec_km": pace_sec_km,
            "Átlagos tempó": pace_str,
            "speed_mps": avg_speed if avg_speed > 0 else np.nan,
            "hr_num": hr_num,
            "Átlagos pulzusszám": hr_num,
            "cad_num": cad_num,
            "Átl. pedálütem": cad_num,
            "power_avg_w": power_avg,
            "power_max_w": power_max,
            "Átl. teljesítmény": power_avg,
            "Max. teljesítmény": power_max,
            "asc_m": asc_m,
            "des_m": np.nan,
            "Teljes emelkedés": asc_m,
            "Teljes süllyedés": np.nan,
            "temp_c": float(a.get("average_temp") or np.nan) if a.get("average_temp") else np.nan,
            "Run_type": run_type_raw,
            "strava_id": a.get("id"),
            "strava_type": sport_type,
            "suffer_score": a.get("suffer_score"),
            "kudos_count": a.get("kudos_count"),
            "vr_num": np.nan,
            "gct_num": np.nan,
            "vo_num": np.nan,
            "stride_num": np.nan,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Dátum").reset_index(drop=True)
    df["up_m_per_km"] = np.where(
        df["dist_km"].notna() & (df["dist_km"] > 0),
        df["asc_m"] / df["dist_km"], np.nan
    )
    df["down_m_per_km"] = np.nan
    df["net_elev_m"] = df["asc_m"]
    df["slope_bucket"] = df["up_m_per_km"].apply(
        lambda u: "flat" if pd.isna(u) or u < 5
        else "rolling" if u < 15
        else "uphill_dominant"
    )
    df["temp_bin"] = df["temp_c"].apply(temp_bin)
    df["Power_fatigue_hint"] = np.where(
        df["power_avg_w"].notna() & df["power_max_w"].notna() & (df["power_avg_w"] > 0),
        df["power_max_w"] / df["power_avg_w"], np.nan
    )
    return df


def get_valid_strava_token(client_id: str, client_secret: str, refresh_tok: str) -> str | None:
    ss = st.session_state
    expires_at = ss.get("strava_token_expires_at", 0)
    now_ts = int(time.time())
    if "strava_access_token" in ss and now_ts < expires_at - 300:
        return ss["strava_access_token"]
    new_tokens = strava_refresh_token(client_id, client_secret, refresh_tok)
    if not new_tokens or "access_token" not in new_tokens:
        return None
    ss["strava_access_token"]      = new_tokens["access_token"]
    ss["strava_token_expires_at"]  = new_tokens["expires_at"]
    if "refresh_token" in new_tokens:
        ss["strava_session_refresh"] = new_tokens["refresh_token"]
    return ss["strava_access_token"]


def render_strava_connect_sidebar():
    client_id, client_secret, refresh_tok = _strava_secrets()
    st.sidebar.divider()
    st.sidebar.header("🟠 Strava kapcsolat")

    if not client_id or not client_secret or not refresh_tok:
        missing = []
        if not client_id:      missing.append("STRAVA_CLIENT_ID")
        if not client_secret:  missing.append("STRAVA_CLIENT_SECRET")
        if not refresh_tok:    missing.append("STRAVA_REFRESH_TOKEN")
        st.sidebar.warning(
            f"Hiányzó Secrets: **{', '.join(missing)}**\n\n"
            "**Lépések:**\n"
            "1. Töltsd le a `get_strava_token.py` scriptet\n"
            "2. Futtasd a saját gépeden → megkapod a refresh_token-t\n"
            "3. Add hozzá a Streamlit Secrets-hez:\n\n"
            "```toml\n"
            "STRAVA_CLIENT_ID     = \"123456\"\n"
            "STRAVA_CLIENT_SECRET = \"abc...\"\n"
            "STRAVA_REFRESH_TOKEN = \"def...\"\n"
            "```"
        )
        return None, "garmin"

    effective_refresh = st.session_state.get("strava_session_refresh", refresh_tok)
    with st.spinner("Strava kapcsolat ellenőrzése…") if "strava_access_token" not in st.session_state else st.sidebar:
        access_token = get_valid_strava_token(client_id, client_secret, effective_refresh)

    if not access_token:
        st.sidebar.error(
            "❌ Strava kapcsolat sikertelen.\n\n"
            "Lehetséges okok:\n"
            "- A STRAVA_REFRESH_TOKEN lejárt vagy érvénytelen\n"
            "- Futtasd újra a `get_strava_token.py` scriptet\n"
            "- Ellenőrizd a Client ID / Secret értékeket"
        )
        return None, "garmin"

    if "strava_athlete_name" not in st.session_state:
        athlete_data = _strava_get("/athlete", access_token)
        if athlete_data:
            st.session_state["strava_athlete_name"] = (
                f"{athlete_data.get('firstname', '')} "
                f"{athlete_data.get('lastname', '')}".strip()
            )

    athlete_name = st.session_state.get("strava_athlete_name", "Strava sportoló")
    st.sidebar.success(f"✅ Kapcsolódva: **{athlete_name}**")

    days_back = st.sidebar.selectbox(
        "Szinkron időszak",
        options=[90, 180, 365, 730],
        index=2,
        format_func=lambda d: {90: "3 hónap", 180: "6 hónap", 365: "1 év", 730: "2 év"}[d],
        key="strava_days_back",
    )

    if st.sidebar.button("🔄 Strava adatok frissítése", key="strava_refresh_btn"):
        fetch_strava_activities.clear()
        st.session_state.pop("strava_access_token", None)
        st.rerun()

    activities = fetch_strava_activities(access_token, days_back=days_back)
    if not activities:
        st.sidebar.warning("Nem találtam futást a megadott időszakban.")
        return None, "strava_empty"

    strava_df = strava_activities_to_df(activities)
    st.sidebar.caption(f"📥 {len(strava_df)} futás szinkronizálva ({days_back} nap)")
    return strava_df, "strava"


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
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    def _conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        v = str(v).strip()
        if v in ("--", "", "None", "nan", "NaN"):
            return np.nan
        v = v.replace(" ", "").replace(",", ".")
        try:
            return float(v)
        except ValueError:
            return np.nan
    return s.apply(_conv).astype(float)


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


# =========================================================
# SLOPE-KORREKCIÓ
# =========================================================
_SLOPE_CORR = {
    "vo_num":   (+0.040,     -0.030),
    "gct_num":  (-0.300,     +0.200),
    "vr_num":   (-0.010,     +0.008),
    "stride_num": (+0.002,   -0.002),
    "cad_num":  (-0.008,     +0.006),
    "hr_num":   (+0.150,     -0.080),
    "pace_sec_km": (-2.5,    +1.8),
}


def slope_correct_series(series, up_m_per_km, down_m_per_km, col_name):
    if col_name not in _SLOPE_CORR:
        return series
    up_c, down_c = _SLOPE_CORR[col_name]
    up = up_m_per_km.fillna(0).clip(lower=0)
    down = down_m_per_km.fillna(0).clip(lower=0)
    correction = up_c * up + down_c * down
    return series - correction


def apply_slope_correction(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if "up_m_per_km" not in out.columns or "down_m_per_km" not in out.columns:
        for c in (cols or list(_SLOPE_CORR.keys())):
            if c in out.columns:
                out[f"{c}_sc"] = out[c]
        return out
    up = out["up_m_per_km"].fillna(0).clip(lower=0)
    down = out["down_m_per_km"].fillna(0).clip(lower=0)
    for c in (cols or list(_SLOPE_CORR.keys())):
        if c not in out.columns:
            continue
        up_c, down_c = _SLOPE_CORR[c]
        correction = up_c * up + down_c * down
        out[f"{c}_sc"] = out[c] - correction
    return out


def slope_correction_summary(up_m_per_km: float, down_m_per_km: float) -> dict:
    up = max(0.0, float(up_m_per_km) if pd.notna(up_m_per_km) else 0.0)
    down = max(0.0, float(down_m_per_km) if pd.notna(down_m_per_km) else 0.0)
    result = {}
    for col, (uc, dc) in _SLOPE_CORR.items():
        corr = uc * up + dc * down
        result[col] = round(corr, 3)
    result["_up_m_per_km"] = up
    result["_down_m_per_km"] = down
    return result


# =========================================================
# ANTROPOMETRIAI SZÁMÍTÁSOK
# =========================================================

def hrmax_from_age(age: int) -> int:
    return round(208 - 0.7 * age)


def age_adjusted_hr_zones(hrmax: int, age: int) -> dict:
    base_zones = {
        "Z1": (0.50, 0.60),
        "Z2": (0.60, 0.70),
        "Z3": (0.70, 0.80),
        "Z4": (0.80, 0.90),
        "Z5": (0.90, 1.00),
    }
    age_shift = max(0.0, (age - 35) * 0.001)
    zones = {}
    for z, (lo, hi) in base_zones.items():
        lo_adj = max(0.45, lo - age_shift)
        hi_adj = max(lo_adj + 0.08, hi - age_shift)
        zones[z] = (round(hrmax * lo_adj), round(hrmax * hi_adj))
    return zones


def fatmax_hr_pct_from_age(age: int) -> tuple[float, float]:
    if age < 30:
        return (0.70, 0.80)
    elif age < 40:
        return (0.68, 0.77)
    elif age < 50:
        return (0.65, 0.75)
    else:
        return (0.62, 0.72)


def recovery_days_age_factor(age: int) -> float:
    if age < 35:
        return 1.0
    elif age < 45:
        return 1.10
    elif age < 55:
        return 1.20
    else:
        return 1.30


def stride_length_ratio(stride_m: float, height_cm: float) -> float:
    if height_cm <= 0 or pd.isna(stride_m) or pd.isna(height_cm):
        return np.nan
    return stride_m / (height_cm / 100.0)


def power_per_kg(power_w: float, weight_kg: float) -> float:
    if weight_kg <= 0 or pd.isna(power_w):
        return np.nan
    return power_w / weight_kg


def bmi(weight_kg: float, height_cm: float) -> float:
    if height_cm <= 0 or weight_kg <= 0:
        return np.nan
    return weight_kg / (height_cm / 100.0) ** 2


# =========================================================
# VERSENYIDŐ BECSLŐ
# =========================================================

def riegel_predict(known_dist_km, known_time_sec, target_dist_km, fatigue_factor=1.0):
    if known_dist_km <= 0 or known_time_sec <= 0 or target_dist_km <= 0:
        return np.nan
    raw = known_time_sec * (target_dist_km / known_dist_km) ** 1.06
    return raw * fatigue_factor


def vo2max_from_race(dist_km: float, time_sec: float) -> float:
    if dist_km <= 0 or time_sec <= 0:
        return np.nan
    t_min = time_sec / 60.0
    v = (dist_km * 1000.0) / t_min
    pct_vo2max = 0.8 + 0.1894393 * np.exp(-0.012778 * t_min) + 0.2989558 * np.exp(-0.1932605 * t_min)
    vo2 = -4.60 + 0.182258 * v + 0.000104 * v ** 2
    return vo2 / pct_vo2max


_RACE_DISTANCES = {
    "1 km":          1.0,
    "1 mérföld":     1.609,
    "3 km":          3.0,
    "5 km":          5.0,
    "10 km":         10.0,
    "15 km":         15.0,
    "Félmaraton":    21.0975,
    "30 km":         30.0,
    "Maraton":       42.195,
    "50 km ultra":   50.0,
    "100 km ultra":  100.0,
}

_RIEGEL_EXP = {
    "1 km":          1.06,
    "1 mérföld":     1.06,
    "3 km":          1.06,
    "5 km":          1.06,
    "10 km":         1.06,
    "15 km":         1.07,
    "Félmaraton":    1.07,
    "30 km":         1.08,
    "Maraton":       1.08,
    "50 km ultra":   1.10,
    "100 km ultra":  1.13,
}


def race_time_table(known_dist, known_time_sec, ctl, atl, fatigue):
    known_km = _RACE_DISTANCES.get(known_dist, 0)
    if known_km <= 0 or known_time_sec <= 0:
        return pd.DataFrame()
    tsb = (float(ctl) - float(atl)) if (ctl and atl) else None
    if tsb is not None:
        if tsb > 10:
            forma = 0.98
            forma_txt = f"🟢 Jó forma (TSB {tsb:+.0f}) → ~2% gyorsabb"
        elif tsb >= 0:
            forma = 1.00
            forma_txt = f"🟡 Semleges (TSB {tsb:+.0f})"
        elif tsb >= -10:
            forma = 1.01
            forma_txt = f"🟠 Kissé fáradt (TSB {tsb:+.0f}) → ~1% lassabb"
        elif tsb >= -25:
            forma = 1.02
            forma_txt = f"🔴 Fáradt (TSB {tsb:+.0f}) → ~2% lassabb"
        else:
            forma = 1.04
            forma_txt = f"🔴 Túlterhelt (TSB {tsb:+.0f}) → ~4% lassabb"
    else:
        forma = 1.0
        forma_txt = "Semleges (nincs TSB adat)"

    rows = []
    for dist_name, dist_km in _RACE_DISTANCES.items():
        exp = _RIEGEL_EXP.get(dist_name, 1.06)
        t_base = known_time_sec * (dist_km / known_km) ** exp
        t_corr = t_base * forma

        def fmt(s):
            if pd.isna(s): return "—"
            s = int(round(s))
            h, r = divmod(s, 3600)
            m, sec = divmod(r, 60)
            return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

        pace_corr = t_corr / dist_km
        rows.append({
            "Táv":              dist_name,
            "Km":               dist_km,
            "Alap becslés":     fmt(t_base),
            "Forma-korrigált":  fmt(t_corr),
            "Tempó (min/km)":   sec_to_pace_str(pace_corr),
            "_t_corr":          t_corr,
        })

    return pd.DataFrame(rows), forma_txt, tsb


# =========================================================
# HŐMÉRSÉKLET-KORREKCIÓ
# =========================================================

def heat_pace_correction(temp_c, hr_avg=None, hrmax=185):
    if pd.isna(temp_c):
        return {"correction_sec": 0, "status": "neutral", "advice": "Nincs hőmérséklet adat."}
    t = float(temp_c)
    if t <= 10:
        corr_sec = 0.0; status = "cold"
    elif t <= 18:
        corr_sec = 0.0; status = "optimal"
    elif t <= 23:
        corr_sec = (t - 18) * 4.0; status = "warm"
    elif t <= 28:
        corr_sec = 20 + (t - 23) * 6.0; status = "hot"
    elif t <= 33:
        corr_sec = 50 + (t - 28) * 8.0; status = "very_hot"
    else:
        corr_sec = 90 + (t - 33) * 10.0; status = "dangerous"

    if t <= 18:
        hr_corr = 0.0
    elif t <= 25:
        hr_corr = (t - 18) * 0.8
    else:
        hr_corr = 5.6 + (t - 25) * 1.5

    advice_map = {
        "cold":      "❄️ Hideg – rövid bemelegítés, kezdj lassabban.",
        "optimal":   "✅ Optimális hőmérséklet – teljes teljesítmény várható.",
        "warm":      f"🟡 Meleg – várható plusz: +{corr_sec:.0f} sec/km. Több folyadék.",
        "hot":       f"🟠 Meleg – várható plusz: +{corr_sec:.0f} sec/km. Lassíts tudatosan.",
        "very_hot":  f"🔴 Nagy meleg – +{corr_sec:.0f} sec/km. Rövidíts, sok folyadék.",
        "dangerous": f"🚨 Veszélyes hőség – +{corr_sec:.0f} sec/km. Fontold meg az elhalasztást.",
    }
    return {
        "temp_c": t, "correction_sec": corr_sec, "hr_correction": hr_corr,
        "status": status, "advice": advice_map[status],
    }


def compute_heat_adjusted_df(df: pd.DataFrame, hr_col: str | None) -> pd.DataFrame:
    out = df.copy()
    if "temp_c" not in out.columns or "pace_sec_km" not in out.columns:
        return out
    corrections = out["temp_c"].apply(lambda t: heat_pace_correction(t))
    out["heat_corr_sec"]     = corrections.apply(lambda d: d["correction_sec"])
    out["heat_hr_corr"]      = corrections.apply(lambda d: d["hr_correction"])
    out["heat_status"]       = corrections.apply(lambda d: d["status"])
    out["pace_heat_adj_sec"] = np.where(
        out["pace_sec_km"].notna() & out["heat_corr_sec"].notna(),
        out["pace_sec_km"] - out["heat_corr_sec"], np.nan,
    )
    if hr_col and hr_col in out.columns:
        out["hr_heat_adj"] = np.where(
            out[hr_col].notna() & out["heat_hr_corr"].notna(),
            out[hr_col] - out["heat_hr_corr"], np.nan,
        )
    return out


def _safe_dropna(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    existing = [c for c in subset if c in df.columns]
    missing  = [c for c in subset if c not in df.columns]
    if missing:
        return df.iloc[0:0].copy()
    return df.dropna(subset=existing)


def find_col(df: pd.DataFrame, *candidates: str) -> str | None:
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
    up = float(up_m_per_km); down = float(down_m_per_km)
    if up < 5 and down < 5: return "flat"
    if max(up, down) < 15: return "rolling"
    if up >= 15 and up > down * 1.2: return "uphill_dominant"
    if down >= 15 and down > up * 1.2: return "downhill_dominant"
    if up >= 15 and down >= 15: return "hilly_mixed"
    return "other"


def duration_to_seconds(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s in ("", "--", "None", "nan"): return np.nan
    s = s.replace(",", ".")
    s_main = s.split(".")[0]
    parts = s_main.split(":")
    try:
        parts = [int(p) for p in parts]
    except Exception:
        return np.nan
    if len(parts) == 3:
        h, m, sec = parts; return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = parts; return m * 60 + sec
    return np.nan


def hr_zone_from_pct(pct):
    if pd.isna(pct): return np.nan
    pct = float(pct)
    if pct < 0.60: return "Z1"
    if pct < 0.70: return "Z2"
    if pct < 0.80: return "Z3"
    if pct < 0.90: return "Z4"
    return "Z5"


def run_type_hu(x):
    mapping = {"easy": "Könnyű", "tempo": "Tempó", "race": "Verseny"}
    if pd.isna(x): return np.nan
    return mapping.get(str(x).strip().lower(), str(x))


def slope_bucket_hu(x):
    mapping = {
        "flat": "Sík", "rolling": "Hullámos", "uphill_dominant": "Emelkedő domináns",
        "downhill_dominant": "Lejtő domináns", "hilly_mixed": "Dombos vegyes", "other": "Egyéb",
    }
    if pd.isna(x): return np.nan
    return mapping.get(str(x).strip(), str(x))


def temp_bin(t):
    if pd.isna(t): return "unk"
    t = float(t)
    if t < 10: return "cold"
    if t < 18: return "cool"
    if t < 24: return "mild"
    return "hot"


def num(v):
    return pd.to_numeric(
        pd.Series([v]).astype(str).str.replace(",", ".", regex=False), errors="coerce",
    ).iloc[0]


def med(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False), errors="coerce",
    ).median()


def _to_minutes(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s in ("--", "", "None", "nan"): return np.nan
    try:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts; return int(h) * 60 + int(m) + int(sec) / 60
        if len(parts) == 2:
            m, sec = parts; return int(m) + int(sec) / 60
    except Exception:
        return np.nan
    return np.nan


# =========================================================
# BASELINE SEGÉDEK
# =========================================================

def get_type_baseline(base_df, last_date, run_type_col, target_type, weeks=12, min_runs=25):
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
    if dfb.empty: return pd.DataFrame()
    start = last_date - pd.Timedelta(weeks=weeks)
    w = dfb[dfb["Dátum"] >= start].copy()
    return w if len(w) >= min_runs else dfb.tail(min_runs).copy()


def get_easy_baseline(base_df, last_date, weeks, min_runs):
    easy = (
        base_df[base_df["Run_type"] == "easy"].copy()
        if "Run_type" in base_df.columns else base_df.copy()
    )
    easy = _safe_dropna(easy, ["Dátum", "Technika_index"]).sort_values("Dátum")
    if pd.isna(last_date): return easy.tail(min_runs).copy()
    start = last_date - pd.Timedelta(weeks=weeks)
    w = easy[(easy["Dátum"] >= start) & (easy["Dátum"] <= last_date)].copy()
    return w if len(w) >= min_runs else easy.tail(min_runs).copy()


# =========================================================
# NAPI COACH ÖSSZEFOGLALÁS
# =========================================================

def daily_coach_summary(base_all, run_type_col, fatigue_col, baseline_weeks, baseline_min_runs):
    if base_all is None or len(base_all) == 0:
        return "ℹ️", "Nincs elég adat a napi összképhez."
    if "Technika_index" not in base_all.columns or base_all["Technika_index"].notna().sum() < 5:
        return "ℹ️", "Technika_index nem elérhető (Strava módban GCT/VO/VR hiányzik). Tölts fel Garmin exportot is a teljes elemzéshez."
    b = _safe_dropna(base_all, ["Dátum", "Technika_index"]).sort_values("Dátum")
    if len(b) < 5: return "ℹ️", "Nincs elég technika adat (legalább ~5 futás kell)."
    easy = (
        b[b[run_type_col] == "easy"].copy()
        if run_type_col and run_type_col in b.columns else b.copy()
    )
    easy = _safe_dropna(easy, ["Dátum", "Technika_index"]).sort_values("Dátum")
    if len(easy) < 5: return "ℹ️", "Kevés easy futás → a napi összkép bizonytalan."
    last_easy = easy.iloc[-1]
    baseline_full = get_easy_baseline(b, last_easy["Dátum"], baseline_weeks, baseline_min_runs)
    if len(baseline_full) < 10: return "ℹ️", "Kevés easy baseline (min ~10) → még gyűjts adatot 1–2 hétig."
    tech_b = float(np.nanmedian(baseline_full["Technika_index"]))
    tech_last = float(last_easy["Technika_index"])
    tech_delta = tech_last - tech_b
    fat_last = fat_b = None
    if (fatigue_col and fatigue_col in b.columns and baseline_full[fatigue_col].notna().sum() >= 8):
        v = last_easy.get(fatigue_col)
        fat_last = float(v) if pd.notna(v) else None
        fat_b = float(np.nanmedian(baseline_full[fatigue_col]))

    tail7 = easy.tail(7)
    tech_trend = (
        float(tail7["Technika_index"].iloc[-1] - tail7["Technika_index"].iloc[0])
        if len(tail7) >= 4 else 0.0
    )

    tc_bad = CFG["daily_coach_tech_bad"]; tc_warn = CFG["daily_coach_tech_warn"]
    fat_bad = CFG["daily_coach_fat_bad"]; fat_warn_v = CFG["daily_coach_fat_warn"]
    tech_bad = tech_delta < tc_bad; tech_warn = tc_bad <= tech_delta < tc_warn
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
# RUNNING ECONOMY
# =========================================================

def compute_efficiency(df, hr_col, pace_col, pwr_col):
    out = df.copy()
    out["_hr"] = to_float_series(out[hr_col]) if hr_col and hr_col in out.columns else np.nan
    if pace_col and pace_col in out.columns:
        out["_pace_sec"] = out[pace_col].apply(pace_str_to_sec)
        out["_speed_mps"] = np.where(
            out["_pace_sec"].notna() & (out["_pace_sec"] > 0), 1000.0 / out["_pace_sec"], np.nan,
        )
    else:
        out["_speed_mps"] = out["speed_mps"] if "speed_mps" in out.columns else np.nan
    out["_pwr"] = to_float_series(out[pwr_col]) if pwr_col and pwr_col in out.columns else np.nan
    out["EF_pace"] = np.where(
        out["_speed_mps"].notna() & out["_hr"].notna() & (out["_hr"] > 0),
        out["_speed_mps"] / out["_hr"], np.nan,
    )
    out["EF_power"] = np.where(
        out["_speed_mps"].notna() & out["_pwr"].notna() & (out["_pwr"] > 0),
        out["_speed_mps"] / out["_pwr"], np.nan,
    )
    out["HR_per_speed"] = np.where(
        out["_speed_mps"].notna() & (out["_speed_mps"] > 0), out["_hr"] / out["_speed_mps"], np.nan,
    )
    out["HR_per_power"] = np.where(
        out["_pwr"].notna() & (out["_pwr"] > 0), out["_hr"] / out["_pwr"], np.nan,
    )
    return out


def estimate_fatmax_from_runs(df_runs, hrmax, hr_col, pace_col, pwr_col=None, min_points=12):
    if df_runs is None or len(df_runs) == 0: return None
    if hr_col not in df_runs.columns or pace_col not in df_runs.columns: return None
    x = compute_efficiency(df_runs, hr_col=hr_col, pace_col=pace_col, pwr_col=pwr_col)
    x = x.dropna(subset=["Dátum", "_hr", "_speed_mps"])
    if len(x) < min_points: return None
    x["HR_pct"] = x["_hr"] / float(hrmax)
    x = x[(x["HR_pct"] >= 0.55) & (x["HR_pct"] <= 0.80)].copy()
    if len(x) < min_points: return None
    bins = np.arange(0.55, 0.80 + 0.0001, 0.02)
    x["HR_bin"] = pd.cut(x["HR_pct"], bins=bins, include_lowest=True)
    use_power = ("EF_power" in x.columns) and (x["EF_power"].notna().sum() >= max(6, min_points // 2))
    ef_col = "EF_power" if use_power else "EF_pace"
    grp = x.groupby("HR_bin")[ef_col].median().dropna()
    if grp.empty: return None
    best_bin = grp.idxmax()
    x_best = x[x["HR_bin"] == best_bin].copy()
    if x_best.empty: return None
    hr_fatmax = float(np.nanmedian(x_best["_hr"]))
    speed_fatmax = float(np.nanmedian(x_best["_speed_mps"]))
    pace_sec = 1000.0 / speed_fatmax if speed_fatmax > 0 else np.nan
    pace_fatmax = sec_to_pace_str(pace_sec)
    pwr_fatmax = (
        float(np.nanmedian(x_best["_pwr"]))
        if (use_power and x_best["_pwr"].notna().any()) else np.nan
    )
    lo = float(best_bin.left) if hasattr(best_bin, "left") else np.nan
    hi = float(best_bin.right) if hasattr(best_bin, "right") else np.nan
    hrpct_mid = (lo + hi) / 2.0 if (pd.notna(lo) and pd.notna(hi)) else np.nan
    return {
        "ef_col": ef_col, "best_bin": str(best_bin), "hrpct_mid": hrpct_mid,
        "hr_fatmax": hr_fatmax, "pace_fatmax": pace_fatmax, "pwr_fatmax": pwr_fatmax,
        "n_points": int(len(x_best)), "table": x, "table_best": x_best,
    }


def aerobic_decoupling_proxy(last_row, baseline_df, hr_col, pace_col, pwr_col):
    if baseline_df is None or len(baseline_df) < 8: return None
    tmp = compute_efficiency(baseline_df, hr_col=hr_col, pace_col=pace_col, pwr_col=pwr_col)
    last_df = compute_efficiency(pd.DataFrame([last_row]), hr_col=hr_col, pace_col=pace_col, pwr_col=pwr_col)
    use_power = (
        "EF_power" in tmp.columns and tmp["EF_power"].notna().sum() >= 8
        and last_df["EF_power"].notna().sum() >= 1
    )
    ef = "EF_power" if use_power else "EF_pace"
    b = float(np.nanmedian(tmp[ef]))
    v = float(last_df[ef].iloc[0]) if pd.notna(last_df[ef].iloc[0]) else np.nan
    if pd.isna(b) or b == 0 or pd.isna(v): return None
    return {"ef": ef, "baseline_med": b, "last": v, "delta_pct": float((v - b) / b * 100.0)}


# =========================================================
# EASY RUN TARGET
# =========================================================

def compute_easy_target(df, hrmax, fatigue_col, run_type_col, hr_col, pace_col, pwr_col,
                         baseline_weeks=12, baseline_min_runs=20):
    if df is None or len(df) == 0: return None
    base = df.dropna(subset=["Dátum"]).sort_values("Dátum").copy()
    if run_type_col and run_type_col in base.columns:
        easy = base[base[run_type_col] == "easy"].copy()
    else:
        easy = base.copy()
    easy = easy.dropna(subset=["Dátum"]).sort_values("Dátum")
    last_date = base["Dátum"].max()
    start_bl = last_date - pd.Timedelta(weeks=baseline_weeks)
    easy_bl = easy[easy["Dátum"] >= start_bl].copy()
    if len(easy_bl) < baseline_min_runs:
        easy_bl = easy.tail(baseline_min_runs).copy()
    if len(easy_bl) < 5: return None

    hr_med = float(np.nanmedian(easy_bl[hr_col])) if hr_col and hr_col in easy_bl.columns else np.nan
    pace_med_sec = float(np.nanmedian(easy_bl["pace_sec_km"])) if "pace_sec_km" in easy_bl.columns else np.nan
    pwr_med = float(np.nanmedian(easy_bl[pwr_col])) if pwr_col and pwr_col in easy_bl.columns else np.nan

    fatmax_hr = np.nan; fatmax_pace_sec = np.nan; fatmax_pwr = np.nan
    if hr_col and "pace_sec_km" in base.columns:
        tmp_fm = easy_bl.copy()
        if "Átlagos tempó" not in tmp_fm.columns and "pace_sec_km" in tmp_fm.columns:
            tmp_fm["_pace_str"] = tmp_fm["pace_sec_km"].apply(
                lambda s: sec_to_pace_str(s) if pd.notna(s) else np.nan
            )
            _pace_col_fm = "_pace_str"
        else:
            _pace_col_fm = pace_col if pace_col else "Átlagos tempó"
        fm = estimate_fatmax_from_runs(tmp_fm, hrmax=hrmax, hr_col=hr_col, pace_col=_pace_col_fm, pwr_col=pwr_col, min_points=10)
        if fm:
            fatmax_hr = fm["hr_fatmax"]
            fatmax_pwr = fm["pwr_fatmax"] if pd.notna(fm["pwr_fatmax"]) else np.nan
            fatmax_pace_sec = pace_str_to_sec(fm["pace_fatmax"])

    if pd.notna(fatmax_hr):
        hr_center = fatmax_hr * 0.94
    elif pd.notna(hr_med):
        hr_center = hr_med
    else:
        hr_center = hrmax * (CFG["easy_target_hr_pct_lo"] + CFG["easy_target_hr_pct_hi"]) / 2

    if pd.notna(fatmax_pace_sec):
        pace_center_sec = fatmax_pace_sec * 1.06
    elif pd.notna(pace_med_sec):
        pace_center_sec = pace_med_sec
    else:
        pace_center_sec = np.nan

    if pd.notna(fatmax_pwr):
        pwr_center = fatmax_pwr * 0.92
    elif pd.notna(pwr_med):
        pwr_center = pwr_med
    else:
        pwr_center = np.nan

    current_fatigue = np.nan; fatigue_penalty_factor = 0.0
    if fatigue_col and fatigue_col in base.columns and base[fatigue_col].notna().any():
        last_run = base.dropna(subset=[fatigue_col]).iloc[-1]
        current_fatigue = float(last_run[fatigue_col])
        fatigue_above_base = max(0.0, current_fatigue - 40.0)
        fatigue_penalty_factor = fatigue_above_base / 10.0

    hr_penalty = CFG["easy_fatigue_penalty_hr"] * fatigue_penalty_factor
    pace_penalty = CFG["easy_fatigue_penalty_pace"] * fatigue_penalty_factor
    pwr_penalty = CFG["easy_fatigue_penalty_pwr"] * fatigue_penalty_factor

    hr_adj = hr_center - hr_penalty
    pace_adj_sec = pace_center_sec + pace_penalty if pd.notna(pace_center_sec) else np.nan
    pwr_adj = pwr_center - pwr_penalty if pd.notna(pwr_center) else np.nan

    band_hr = CFG["easy_target_band_hr"]; band_pace = CFG["easy_target_band_pace"]; band_pwr = CFG["easy_target_band_pwr"]
    hr_lo = round(hr_adj - band_hr); hr_hi = round(hr_adj + band_hr)
    pace_lo_sec = pace_adj_sec - band_pace if pd.notna(pace_adj_sec) else np.nan
    pace_hi_sec = pace_adj_sec + band_pace if pd.notna(pace_adj_sec) else np.nan
    pwr_lo = round(pwr_adj - band_pwr) if pd.notna(pwr_adj) else np.nan
    pwr_hi = round(pwr_adj + band_pwr) if pd.notna(pwr_adj) else np.nan

    hr_lo = max(hr_lo, round(hrmax * CFG["easy_target_hr_pct_lo"]))
    hr_hi = min(hr_hi, round(hrmax * CFG["easy_target_hr_pct_hi"]))

    if pd.notna(current_fatigue):
        if current_fatigue >= 70:
            readiness_status = "🔴"; readiness_text = "Nagy fáradtság – nagyon konzervatív easy, vagy pihenőnap"
        elif current_fatigue >= 50:
            readiness_status = "🟠"; readiness_text = "Közepes fáradtság – legyen valóban könnyű, ne csábulj gyorsabb tempóra"
        elif current_fatigue >= 30:
            readiness_status = "🟡"; readiness_text = "Normál állapot – klasszikus easy, a zónát tartsd be"
        else:
            readiness_status = "🟢"; readiness_text = "Friss állapot – a zóna betartásával akár kicsit több km is mehet"
    else:
        readiness_status = "⚪"; readiness_text = "Fatigue adat nélkül ez a személyes easy medián alapján számolt zóna"

    return {
        "hr_lo": hr_lo, "hr_hi": hr_hi,
        "pace_lo": sec_to_pace_str(pace_lo_sec), "pace_hi": sec_to_pace_str(pace_hi_sec),
        "pwr_lo": pwr_lo, "pwr_hi": pwr_hi,
        "hr_center": round(hr_adj, 1), "pace_center": sec_to_pace_str(pace_adj_sec),
        "pwr_center": round(pwr_adj, 1) if pd.notna(pwr_adj) else np.nan,
        "fatmax_hr": fatmax_hr, "fatmax_pace": sec_to_pace_str(fatmax_pace_sec), "fatmax_pwr": fatmax_pwr,
        "hr_med_easy": hr_med, "pace_med_easy": sec_to_pace_str(pace_med_sec), "pwr_med_easy": pwr_med,
        "current_fatigue": current_fatigue, "fatigue_penalty_factor": round(fatigue_penalty_factor, 2),
        "hr_penalty": round(hr_penalty, 1), "pace_penalty_sec": round(pace_penalty, 1),
        "pwr_penalty": round(pwr_penalty, 1),
        "readiness_status": readiness_status, "readiness_text": readiness_text,
        "easy_history": easy_bl.tail(30)[
            [c for c in ["Dátum", hr_col, "pace_sec_km", pwr_col, fatigue_col]
             if c and c in easy_bl.columns]
        ].copy() if len(easy_bl) >= 5 else pd.DataFrame(),
    }


# =========================================================
# TSS PROXY
# =========================================================

def compute_tss_proxy(df: pd.DataFrame, hrmax: int) -> pd.Series:
    hr_thr = 0.85 * hrmax
    tss = np.full(len(df), np.nan)
    has_hr = df["hr_num"].notna() & df["dur_sec"].notna() & (df["dur_sec"] > 0)
    if has_hr.any():
        IF_ = df.loc[has_hr, "hr_num"] / hr_thr
        tss_hr = (df.loc[has_hr, "dur_sec"] * df.loc[has_hr, "hr_num"] * IF_) / (hrmax * 3600) * 100
        tss[np.where(has_hr)[0]] = tss_hr.values
    no_hr = ~has_hr & df["pace_sec_km"].notna() & df["dist_km"].notna()
    if no_hr.any():
        valid_pace = df.loc[df["pace_sec_km"].notna(), "pace_sec_km"]
        thr_pace = float(np.nanpercentile(valid_pace, 40)) if len(valid_pace) >= 10 else np.nan
        if pd.notna(thr_pace) and thr_pace > 0:
            IF_pace = thr_pace / df.loc[no_hr, "pace_sec_km"]
            tss_pace = IF_pace ** 2 * df.loc[no_hr, "dist_km"] * (thr_pace / 60)
            tss[np.where(no_hr)[0]] = tss_pace.clip(0, 500).values
    return pd.Series(tss, index=df.index)


def compute_acwr(df, load_col="TSS_proxy", acute_days=7, chronic_days=28):
    if load_col not in df.columns or df[load_col].isna().all():
        return pd.DataFrame()
    daily = (
        df.dropna(subset=["Dátum"])
        .groupby(df["Dátum"].dt.date)[load_col].sum()
        .reset_index().rename(columns={"Dátum": "date", load_col: "load"})
        .sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
    daily.columns = ["date", "load"]
    daily["acute"] = daily["load"].rolling(acute_days, min_periods=1).mean()
    daily["chronic"] = daily["load"].rolling(chronic_days, min_periods=7).mean()
    daily["acwr"] = np.where(
        daily["chronic"].notna() & (daily["chronic"] > 0), daily["acute"] / daily["chronic"], np.nan,
    )
    def _acwr_status(v):
        if pd.isna(v): return "ismeretlen"
        if v < CFG["acwr_safe_lo"]: return "alulterhelt"
        if v <= CFG["acwr_safe_hi"]: return "optimális"
        if v <= CFG["acwr_warn_hi"]: return "figyelmeztető"
        return "veszélyes"
    daily["acwr_status"] = daily["acwr"].apply(_acwr_status)
    return daily


def compute_ctl_atl_tsb(daily_load):
    if daily_load.empty or "load" not in daily_load.columns:
        return pd.DataFrame()
    alpha_ctl = 1.0 - np.exp(-1.0 / CFG["ctl_decay"])
    alpha_atl = 1.0 - np.exp(-1.0 / CFG["atl_decay"])
    loads = daily_load["load"].fillna(0).to_numpy(dtype=float)
    ctl = np.zeros(len(loads)); atl = np.zeros(len(loads))
    ctl[0] = loads[0]; atl[0] = loads[0]
    for i in range(1, len(loads)):
        ctl[i] = ctl[i-1] + alpha_ctl * (loads[i] - ctl[i-1])
        atl[i] = atl[i-1] + alpha_atl * (loads[i] - atl[i-1])
    out = daily_load.copy()
    out["CTL"] = ctl; out["ATL"] = atl; out["TSB"] = ctl - atl
    def _tsb_status(v):
        if pd.isna(v): return "ismeretlen"
        if v >= CFG["tsb_green"]: return "forma"
        if v >= -10: return "friss"
        if v >= -20: return "fáradt"
        if v >= CFG["tsb_red"]: return "túlterhelt"
        return "kritikus"
    out["TSB_status"] = out["TSB"].apply(_tsb_status)
    return out


def injury_risk_score(acwr_val, tsb_val, fatigue_val, asym_val, ramp_val):
    components = []; explanations = []
    if acwr_val is not None and not np.isnan(float(acwr_val)):
        v = float(acwr_val)
        if v < CFG["acwr_safe_lo"]: r, txt = 20, f"ACWR {v:.2f}: alulterhelt"
        elif v <= CFG["acwr_safe_hi"]: r, txt = 5, f"ACWR {v:.2f}: optimális ✓"
        elif v <= CFG["acwr_warn_hi"]: r, txt = 50, f"ACWR {v:.2f}: figyelmeztető"
        else: r, txt = 90, f"ACWR {v:.2f}: veszélyes"
        components.append((r, 0.35)); explanations.append(txt)
    if tsb_val is not None and not np.isnan(float(tsb_val)):
        v = float(tsb_val)
        if v >= CFG["tsb_green"]: r, txt = 10, f"TSB {v:+.1f}: jó forma ✓"
        elif v >= -10: r, txt = 25, f"TSB {v:+.1f}: friss"
        elif v >= -20: r, txt = 55, f"TSB {v:+.1f}: fáradt"
        elif v >= CFG["tsb_red"]: r, txt = 80, f"TSB {v:+.1f}: túlterhelt"
        else: r, txt = 95, f"TSB {v:+.1f}: kritikus"
        components.append((r, 0.25)); explanations.append(txt)
    if fatigue_val is not None and not np.isnan(float(fatigue_val)):
        v = float(fatigue_val)
        r = float(np.clip(v, 0, 100))
        txt = f"Fatigue {v:.0f}: {'magas' if v > 70 else 'közepes' if v > 50 else 'alacsony ✓'}"
        components.append((r, 0.20)); explanations.append(txt)
    if ramp_val is not None and not np.isnan(float(ramp_val)):
        v = float(ramp_val)
        r = 10 if v <= 8 else 45 if v <= 12 else 75 if v <= 20 else 95
        if v > 8: explanations.append(f"Ramp rate {v:+.1f}%: gyors terhelésnövelés")
        components.append((r, 0.15))
    if asym_val is not None and not np.isnan(float(asym_val)):
        v = float(asym_val)
        if v < CFG["asym_warn_pct"]: r = 5
        elif v < CFG["asym_red_pct"]: r = 40; explanations.append(f"Aszimmetria {v:.1f}%: figyelemre méltó")
        else: r = 75; explanations.append(f"Aszimmetria {v:.1f}%: magas")
        components.append((r, 0.05))
    if not components: return np.nan, "ismeretlen", ["Nincs elég adat."]
    total_w = sum(w for _, w in components)
    score = float(np.clip(sum(r * w for r, w in components) / total_w, 0, 100))
    status = (
        "🟢 Alacsony" if score < 25 else "🟡 Közepes" if score < 50
        else "🟠 Magas" if score < 70 else "🔴 Kritikus"
    )
    return score, status, [e for e in explanations if e is not None]


# =========================================================
# RECOVERY TIME MODELL
# =========================================================

def compute_recovery_model(df, fatigue_col, run_type_col, lookback_weeks=24, min_events=8):
    if "Technika_index" not in df.columns or fatigue_col not in df.columns: return None
    base = _safe_dropna(df, ["Dátum", "Technika_index", fatigue_col]).sort_values("Dátum").copy()
    if len(base) < min_events * 3: return None
    cutoff = base["Dátum"].max() - pd.Timedelta(weeks=lookback_weeks)
    base = base[base["Dátum"] >= cutoff].copy()
    easy = (
        base[base[run_type_col] == "easy"].copy()
        if run_type_col and run_type_col in base.columns else base.copy()
    )
    if len(easy) < min_events: return None
    events = base[base[fatigue_col] > 65].copy()
    if len(events) < min_events:
        thr = np.nanpercentile(base[fatigue_col], 75)
        events = base[base[fatigue_col] >= thr].copy()
    if len(events) < 4: return None
    rec_days_list, fat_list, pre_tech_list = [], [], []
    for _, ev in events.iterrows():
        ev_date = ev["Dátum"]; fat_val = float(ev[fatigue_col])
        pre_w = easy[
            (easy["Dátum"] >= ev_date - pd.Timedelta(days=21)) &
            (easy["Dátum"] < ev_date - pd.Timedelta(days=1))
        ]
        if len(pre_w) < 2: continue
        pre_tech = float(np.nanmedian(pre_w["Technika_index"]))
        post = easy[
            (easy["Dátum"] > ev_date) & (easy["Dátum"] <= ev_date + pd.Timedelta(days=30))
        ].sort_values("Dátum")
        if len(post) < 2: continue
        recovered_day = None
        for i in range(len(post) - 1):
            t1 = float(post.iloc[i]["Technika_index"]); t2 = float(post.iloc[i+1]["Technika_index"])
            if abs(t1 - pre_tech) <= 7 and abs(t2 - pre_tech) <= 7:
                recovered_day = (post.iloc[i]["Dátum"] - ev_date).days; break
        if recovered_day is not None and 0 < recovered_day <= 30:
            rec_days_list.append(recovered_day); fat_list.append(fat_val); pre_tech_list.append(pre_tech)
    if len(rec_days_list) < 4: return None
    rec_arr = np.array(rec_days_list, dtype=float); fat_arr = np.array(fat_list, dtype=float)
    coeffs = np.polyfit(fat_arr, rec_arr, 1) if np.std(fat_arr) > 1e-6 else [0, float(np.median(rec_arr))]
    current_fat = None
    if df[fatigue_col].notna().any():
        current_fat = float(df.dropna(subset=[fatigue_col]).sort_values("Dátum").iloc[-1][fatigue_col])
    predicted_days = None
    if current_fat is not None:
        predicted_days = max(0, int(round(float(np.polyval(coeffs, current_fat)))))
    return {
        "rec_arr": rec_arr, "fat_arr": fat_arr, "coeffs": coeffs,
        "median_recovery": float(np.median(rec_arr)), "n_events": len(rec_arr),
        "current_fat": current_fat, "predicted_days": predicted_days,
        "df": pd.DataFrame({"fatigue": fat_arr, "recovery_days": rec_arr}),
    }


# =========================================================
# ASZIMMETRIA
# =========================================================

def compute_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    def _asym_pct(left_s, right_s):
        mean_lr = (left_s + right_s) / 2
        return np.where(
            left_s.notna() & right_s.notna() & (mean_lr > 0),
            np.abs(left_s - right_s) / mean_lr * 100, np.nan,
        )
    lc = next((c for c in df.columns if "bal" in c.lower() and "gct" in c.lower()), None)
    rc = next((c for c in df.columns if "jobb" in c.lower() and "gct" in c.lower()), None)
    result["asym_gct_pct"] = (
        pd.Series(_asym_pct(to_float_series(df[lc]), to_float_series(df[rc])), index=df.index)
        if lc and rc else np.nan
    )
    lc = next((c for c in df.columns if "bal" in c.lower() and ("lépés" in c.lower() or "stride" in c.lower())), None)
    rc = next((c for c in df.columns if "jobb" in c.lower() and ("lépés" in c.lower() or "stride" in c.lower())), None)
    result["asym_stride_pct"] = (
        pd.Series(_asym_pct(to_float_series(df[lc]), to_float_series(df[rc])), index=df.index)
        if lc and rc else np.nan
    )
    bal_col = next((c for c in df.columns if "egyensúly" in c.lower() or "balance" in c.lower()), None)
    result["asym_power_pct"] = (
        np.abs(to_float_series(df[bal_col]) - 50.0) * 2 if bal_col else np.nan
    )
    asym_w = {"asym_gct_pct": 0.50, "asym_stride_pct": 0.30, "asym_power_pct": 0.20}
    available = [c for c in asym_w if c in result.columns and result[c].notna().any()]
    if available:
        total_w = sum(asym_w[c] for c in available)
        result["Asymmetry_score"] = (
            sum(result[c].fillna(0) * asym_w[c] for c in available) / total_w
        ).clip(0, 20)
    else:
        result["Asymmetry_score"] = np.nan
    return result


# =========================================================
# OPTIMÁLIS TERHELÉSI ABLAK
# =========================================================

def compute_optimal_load_window(df, load_col, tech_col="Technika_index", res_col="RES_plus", lag_weeks=2):
    if load_col not in df.columns or tech_col not in df.columns: return None
    w = df.dropna(subset=["Dátum"]).copy()
    w["week"] = w["Dátum"].dt.to_period("W").dt.start_time
    agg_spec: dict = {"load": (load_col, "sum")}
    if tech_col in w.columns: agg_spec["tech"] = (tech_col, "mean")
    if res_col in w.columns and w[res_col].notna().any(): agg_spec["res"] = (res_col, "mean")
    weekly = w.groupby("week", as_index=False).agg(**agg_spec).sort_values("week")
    if len(weekly) < CFG["optload_min_weeks"]: return None
    if "tech" in weekly.columns:
        weekly["tech_future"] = weekly["tech"].shift(-lag_weeks).rolling(lag_weeks, min_periods=1).mean()
    if "res" in weekly.columns:
        weekly["res_future"] = weekly["res"].shift(-lag_weeks).rolling(lag_weeks, min_periods=1).mean()
    out_cols = [c for c in ["tech_future", "res_future"] if c in weekly.columns and weekly[c].notna().sum() >= 8]
    if not out_cols: return None
    weekly["outcome"] = weekly[out_cols].mean(axis=1)
    weekly = weekly.dropna(subset=["outcome", "load"])
    if len(weekly) < 8: return None
    weekly["load_bin"] = pd.qcut(weekly["load"], q=5, duplicates="drop")
    bin_means = (
        weekly.groupby("load_bin", observed=True)
        .agg(load_mid=("load", "median"), outcome_med=("outcome", "median"), count=("load", "count"))
        .reset_index().sort_values("load_mid")
    )
    best_idx = bin_means["outcome_med"].idxmax()
    best_bin = bin_means.loc[best_idx]
    corr = float(weekly["load"].corr(weekly["outcome"]))
    lo = float(best_bin["load_bin"].left) if hasattr(best_bin["load_bin"], "left") else np.nan
    hi = float(best_bin["load_bin"].right) if hasattr(best_bin["load_bin"], "right") else np.nan
    return {
        "weekly": weekly, "bin_means": bin_means, "optimum_lo": lo, "optimum_hi": hi,
        "optimum_mid": float(best_bin["load_mid"]), "corr": corr, "lag_weeks": lag_weeks,
        "load_col": load_col, "n_weeks": len(weekly),
    }


def compute_intensity_ratio_effect(df, run_type_col, tech_col="Technika_index", lag_weeks=3):
    if run_type_col is None or run_type_col not in df.columns or tech_col not in df.columns: return None
    w = df.dropna(subset=["Dátum", run_type_col]).copy()
    w["week"] = w["Dátum"].dt.to_period("W").dt.start_time
    type_counts = (
        w.groupby(["week", run_type_col], observed=True).size().unstack(fill_value=0).reset_index()
    )
    for t in ["easy", "tempo", "race", "interval"]:
        if t not in type_counts.columns: type_counts[t] = 0
    type_counts["total"] = type_counts[["easy", "tempo", "race", "interval"]].sum(axis=1)
    for t in ["easy", "tempo", "race", "interval"]:
        type_counts[f"{t}_pct"] = np.where(
            type_counts["total"] > 0, type_counts[t] / type_counts["total"] * 100, np.nan
        )
    tech_weekly = w.groupby("week")[tech_col].mean().reset_index()
    merged = type_counts.merge(tech_weekly, on="week", how="inner").sort_values("week")
    if len(merged) < 10: return None
    merged["tech_future"] = merged[tech_col].shift(-lag_weeks).rolling(lag_weeks, min_periods=1).mean()
    merged = merged.dropna(subset=["tech_future"])
    if len(merged) < 8: return None
    corrs = {}
    for t in ["easy", "tempo", "race", "interval"]:
        col = f"{t}_pct"
        if col in merged.columns and merged[col].notna().sum() >= 8:
            c = float(merged[col].corr(merged["tech_future"]))
            if not np.isnan(c): corrs[t] = c
    if not corrs: return None
    best_type = max(corrs, key=lambda k: corrs[k])
    best_pct_col = f"{best_type}_pct"
    mc = merged.dropna(subset=[best_pct_col, "tech_future"])
    opt_lo = opt_hi = np.nan
    if len(mc) >= 8:
        mc = mc.copy()
        mc["pct_bin"] = pd.qcut(mc[best_pct_col], q=4, duplicates="drop")
        best_pct_bin = mc.groupby("pct_bin", observed=True)["tech_future"].median().idxmax()
        opt_lo = float(best_pct_bin.left) if hasattr(best_pct_bin, "left") else np.nan
        opt_hi = float(best_pct_bin.right) if hasattr(best_pct_bin, "right") else np.nan
    return {
        "corrs": corrs, "best_type": best_type, "opt_lo": opt_lo, "opt_hi": opt_hi,
        "merged": merged, "lag_weeks": lag_weeks,
    }


# =========================================================
# ADATBEOLVASÁS
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
            text = file_bytes.decode(enc); break
        except Exception:
            continue
    if text is None: text = file_bytes.decode("utf-8", errors="replace")
    lines = text.strip().splitlines()
    if not lines: return pd.DataFrame()
    header = lines[0].split(",")
    data = []
    for raw_data in lines[1:]:
        raw_data = raw_data.replace('""', '"')
        if raw_data.startswith('"'): raw_data = raw_data[1:]
        reader = csv.reader([raw_data], delimiter=",", quotechar='"')
        row = next(reader); data.append(row)
    df = pd.DataFrame(data, columns=header)
    df.columns = pd.Index(df.columns).astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def fix_mojibake_columns(df: pd.DataFrame) -> pd.DataFrame:
    if any("Ă" in c for c in df.columns.astype(str)):
        new_cols = []
        for c in df.columns.astype(str):
            try: new_cols.append(c.encode("latin1").decode("utf-8"))
            except Exception: new_cols.append(c)
        df = df.copy(); df.columns = new_cols
    return df


# =========================================================
# TELJES PIPELINE
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
def full_pipeline(file_bytes: bytes, file_name: str, _v: int = 3) -> pd.DataFrame:
    df0 = load_any(file_bytes, file_name)
    df0 = fix_mojibake_columns(df0)
    df = df0.copy()

    date_candidates = [c for c in df.columns if any(k in c.lower() for k in ["dátum", "datum", "date"])]
    if date_candidates:
        s = df[date_candidates[0]].astype(str).str.strip().replace({"--": np.nan, "": np.nan, "None": np.nan})
        dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S")
        if dt.notna().sum() == 0: dt = pd.to_datetime(s, errors="coerce")
        df["Dátum"] = dt
    else:
        df["Dátum"] = pd.NaT

    mask_run = (
        df["Tevékenység típusa"].astype(str).str.contains("Fut", na=False)
        if "Tevékenység típusa" in df.columns else pd.Series(True, index=df.index)
    )

    for src, dst in NUM_MAP.items():
        df[dst] = to_float_series(df[src]) if src in df.columns else np.nan

    df["pace_sec_km"] = (
        df["Átlagos tempó"].apply(pace_to_sec_per_km) if "Átlagos tempó" in df.columns else np.nan
    )
    df["speed_mps"] = np.where(
        df["pace_sec_km"].notna() & (df["pace_sec_km"] > 0), 1000.0 / df["pace_sec_km"], np.nan,
    )
    df["up_m_per_km"] = np.where(df["dist_km"].notna() & (df["dist_km"] > 0), df["asc_m"] / df["dist_km"], np.nan)
    df["down_m_per_km"] = np.where(df["dist_km"].notna() & (df["dist_km"] > 0), df["des_m"] / df["dist_km"], np.nan)
    df["net_elev_m"] = df["asc_m"] - df["des_m"]
    df["slope_bucket"] = [slope_bucket_row(u, d2) for u, d2 in zip(df["up_m_per_km"], df["down_m_per_km"])]

    df["Run_type"] = df["Cím"].apply(classify_from_title) if "Cím" in df.columns else None
    valid_pace = df.loc[mask_run & df["pace_sec_km"].notna(), "pace_sec_km"]
    if valid_pace.notna().sum() >= 30:
        p40 = np.nanpercentile(valid_pace, 40); p80 = np.nanpercentile(valid_pace, 80)
        def classify_from_pace(row):
            if pd.notna(row["Run_type"]): return row["Run_type"]
            ps = row["pace_sec_km"]
            if pd.isna(ps): return None
            if ps >= p80: return "easy"
            if ps >= p40: return "tempo"
            return "race"
        df["Run_type"] = df.apply(classify_from_pace, axis=1)

    if "Run_type" in df.columns:
        df["Run_type"] = (
            df["Run_type"].astype(str).str.strip().str.lower()
            .replace({"none": np.nan, "nan": np.nan, "": np.nan})
        )

    df["power_avg_w"] = to_float_series(df["Átl. teljesítmény"]) if "Átl. teljesítmény" in df.columns else np.nan
    df["power_max_w"] = to_float_series(df["Max. teljesítmény"]) if "Max. teljesítmény" in df.columns else np.nan
    df["Power_fatigue_hint"] = np.where(
        df["power_avg_w"].notna() & df["power_max_w"].notna() & (df["power_avg_w"] > 0),
        df["power_max_w"] / df["power_avg_w"], np.nan,
    )
    interval_mask = (
        df["Power_fatigue_hint"].notna() & (df["Power_fatigue_hint"] >= 2.0)
        & df["Run_type"].isin(["easy", "tempo", np.nan, None])
    )
    df.loc[interval_mask, "Run_type"] = "interval"

    temp_col = find_col(df, "Hőmérséklet", "Temperature", "Temp", "Átlag hőmérséklet", "Avg Temperature")
    df["temp_c"] = to_float_series(df[temp_col]) if temp_col else np.nan
    df["temp_bin"] = df["temp_c"].apply(temp_bin)
    df["hr_avg"] = df["hr_num"] if "hr_num" in df.columns else np.nan
    df["hr_per_watt"] = np.where(
        df["hr_avg"].notna() & df["power_avg_w"].notna() & (df["power_avg_w"] > 0),
        df["hr_avg"] / df["power_avg_w"], np.nan,
    )
    df["power_per_speed"] = np.where(
        df["power_avg_w"].notna() & df["speed_mps"].notna() & (df["speed_mps"] > 0),
        df["power_avg_w"] / df["speed_mps"], np.nan,
    )

    # Technika_index
    df["Technika_index"] = np.nan
    tech_base = df[mask_run & df["speed_mps"].notna()].copy()
    if len(tech_base) >= CFG["tech_min_total"]:
        tech_base = apply_slope_correction(tech_base, cols=["vr_num", "gct_num", "vo_num", "cad_num", "stride_num"])
        _vr   = "vr_num_sc"   if "vr_num_sc"   in tech_base.columns else "vr_num"
        _gct  = "gct_num_sc"  if "gct_num_sc"  in tech_base.columns else "gct_num"
        _vo   = "vo_num_sc"   if "vo_num_sc"   in tech_base.columns else "vo_num"
        _cad  = "cad_num_sc"  if "cad_num_sc"  in tech_base.columns else "cad_num"
        _str  = "stride_num_sc" if "stride_num_sc" in tech_base.columns else "stride_num"
        tech_base["speed_bin"] = pd.qcut(tech_base["speed_mps"], q=CFG["speed_bins"], duplicates="drop")
        for col in ["skill_vr", "skill_gct", "skill_vo", "skill_cad", "skill_stride"]:
            tech_base[col] = np.nan

        def _fill_tech_group(g, target):
            idx = g.index; min_n = CFG["tech_min_group"]
            if g[_vr].notna().sum() >= min_n: target.loc[idx, "skill_vr"] = -robust_z(g[_vr], g[_vr])
            if g[_gct].notna().sum() >= min_n: target.loc[idx, "skill_gct"] = -robust_z(g[_gct], g[_gct])
            if g[_vo].notna().sum() >= min_n: target.loc[idx, "skill_vo"] = -robust_z(g[_vo], g[_vo])
            if g[_cad].notna().sum() >= min_n: target.loc[idx, "skill_cad"] = -np.abs(robust_z(g[_cad], g[_cad]))
            if g[_str].notna().sum() >= min_n: target.loc[idx, "skill_stride"] = robust_z(g[_str], g[_str])

        grouped = tech_base.groupby(["speed_bin", "slope_bucket"], dropna=False)
        for _, g in grouped:
            if len(g) >= CFG["tech_min_group"]: _fill_tech_group(g, tech_base)

        still_nan = (
            tech_base["skill_vr"].isna() & tech_base["skill_gct"].isna()
            & tech_base["skill_vo"].isna() & tech_base["skill_cad"].isna()
        )
        if still_nan.any():
            for _, g in tech_base[still_nan].groupby("speed_bin"):
                _fill_tech_group(g, tech_base)

        w = CFG["tech_weights"]
        raw = (
            w["vr"] * tech_base["skill_vr"].fillna(0) + w["gct"] * tech_base["skill_gct"].fillna(0)
            + w["vo"] * tech_base["skill_vo"].fillna(0) + w["cad"] * tech_base["skill_cad"].fillna(0)
            + w["stride"] * tech_base["skill_stride"].fillna(0)
        )
        p5, p95 = np.nanpercentile(raw, 5), np.nanpercentile(raw, 95)
        tech_base["Technika_index"] = (100 * (raw - p5) / (p95 - p5 + 1e-9)).clip(0, 100)
        df.loc[tech_base.index, "Technika_index"] = tech_base["Technika_index"].values

    # Fatigue_score
    df["Fatigue_score"] = np.nan
    df["Fatigue_flag"]  = np.nan
    df["Fatigue_type"]  = pd.array([None] * len(df), dtype=object)
    fat = df[mask_run & df["Technika_index"].notna()].copy()
    fat = apply_slope_correction(fat, cols=["gct_num", "vr_num", "cad_num", "hr_num", "pace_sec_km"])
    _fat_gct  = "gct_num_sc"     if "gct_num_sc"     in fat.columns else "gct_num"
    _fat_vr   = "vr_num_sc"      if "vr_num_sc"      in fat.columns else "vr_num"
    _fat_cad  = "cad_num_sc"     if "cad_num_sc"     in fat.columns else "cad_num"
    _fat_hr   = "hr_num_sc"      if "hr_num_sc"      in fat.columns else "hr_num"
    _fat_pace = "pace_sec_km_sc" if "pace_sec_km_sc" in fat.columns else "pace_sec_km"
    fat["hr_per_pace"] = np.where(
        fat[_fat_hr].notna() & fat[_fat_pace].notna() & (fat[_fat_pace] > 0),
        fat[_fat_hr] / fat[_fat_pace], np.nan,
    )

    if len(fat) >= CFG["fat_min_total"]:
        easy_base = fat[fat["Run_type"] == "easy"].copy() if "Run_type" in fat.columns else fat.copy()

        def _compute_fat_z_vectorized(fat_df, easy_df):
            result = fat_df.copy()
            for col in ["fatigue_gct", "fatigue_vr", "fatigue_cad", "fatigue_hr"]:
                result[col] = 0.0
            def _get_base(slope_val):
                if len(easy_df) >= CFG["fat_easy_full"] and pd.notna(slope_val):
                    sub = easy_df[easy_df["slope_bucket"] == slope_val]
                    if len(sub) >= 20: return sub
                if len(easy_df) >= CFG["fat_easy_min"]: return easy_df
                return fat_df
            for sb, group in fat_df.groupby("slope_bucket", dropna=False):
                base = _get_base(sb); idx = group.index
                for fat_col, src_col in [("fatigue_gct", _fat_gct), ("fatigue_vr", _fat_vr), ("fatigue_hr", "hr_per_pace")]:
                    if group[src_col].notna().sum() >= 5 and base[src_col].notna().sum() >= CFG["fat_min_baseline"]:
                        z = robust_z(group[src_col], base[src_col])
                        result.loc[idx, fat_col] = z.fillna(0).values
                if group[_fat_cad].notna().sum() >= 5 and base[_fat_cad].notna().sum() >= CFG["fat_min_baseline"]:
                    z = robust_z(group[_fat_cad], base[_fat_cad])
                    result.loc[idx, "fatigue_cad"] = np.abs(z.fillna(0).values)
            return result

        fat = _compute_fat_z_vectorized(fat, easy_base)
        fw = CFG["fat_weights"]
        raw_f = (
            fw["gct"] * fat["fatigue_gct"] + fw["vr"] * fat["fatigue_vr"]
            + fw["cad"] * fat["fatigue_cad"] + fw["hr"] * fat["fatigue_hr"]
        )
        p5, p95 = np.nanpercentile(raw_f, 5), np.nanpercentile(raw_f, 95)
        fat["Fatigue_score"] = (100 * (raw_f - p5) / (p95 - p5 + 1e-9)).clip(0, 100)
        fat["Fatigue_flag"] = (fat["Fatigue_score"] > 65).astype(float)
        mech = fat["fatigue_gct"] + fat["fatigue_vr"]; cardio = fat["fatigue_hr"]
        conditions = [(mech > 1.5) & (cardio < 0.5), (cardio > 1.2) & (mech < 1.0), (mech > 1.2) & (cardio > 1.0)]
        choices = ["mechanical", "cardio", "mixed"]
        fat["Fatigue_type"] = np.select(conditions, choices, default="none")
        df.loc[fat.index, "Fatigue_score"] = fat["Fatigue_score"].values
        df.loc[fat.index, "Fatigue_flag"] = fat["Fatigue_flag"].values
        df.loc[fat.index, "Fatigue_type"] = fat["Fatigue_type"].astype(object).values

    # RES+
    df["RES_plus"] = np.nan
    eco = df[mask_run].copy(); eco = eco.dropna(subset=["Dátum", "speed_mps"])
    if len(eco) >= CFG["res_min_total"]:
        eco = apply_slope_correction(eco, cols=["gct_num", "vr_num", "vo_num", "cad_num", "stride_num"])
        _eco_gct    = "gct_num_sc"    if "gct_num_sc"    in eco.columns else "gct_num"
        _eco_vr     = "vr_num_sc"     if "vr_num_sc"     in eco.columns else "vr_num"
        _eco_vo     = "vo_num_sc"     if "vo_num_sc"     in eco.columns else "vo_num"
        _eco_cad    = "cad_num_sc"    if "cad_num_sc"    in eco.columns else "cad_num"
        _eco_stride = "stride_num_sc" if "stride_num_sc" in eco.columns else "stride_num"
        eco["speed_bin"] = pd.qcut(eco["speed_mps"], q=CFG["speed_bins"], duplicates="drop")
        group_cols = ["speed_bin", "slope_bucket", "temp_bin"]
        for c in ["eco_hrpw", "eco_pps", "eco_gct", "eco_vr", "eco_vo", "eco_cad", "eco_stride"]:
            eco[c] = np.nan

        def _fill_eco_group(g, target):
            idx = g.index; min_n = CFG["res_min_group"]
            if g["hr_per_watt"].notna().sum() >= min_n: target.loc[idx, "eco_hrpw"] = -robust_z(g["hr_per_watt"], g["hr_per_watt"])
            if g["power_per_speed"].notna().sum() >= min_n: target.loc[idx, "eco_pps"] = -robust_z(g["power_per_speed"], g["power_per_speed"])
            if g[_eco_gct].notna().sum() >= min_n: target.loc[idx, "eco_gct"] = -robust_z(g[_eco_gct], g[_eco_gct])
            if g[_eco_vr].notna().sum() >= min_n: target.loc[idx, "eco_vr"] = -robust_z(g[_eco_vr], g[_eco_vr])
            if g[_eco_vo].notna().sum() >= min_n: target.loc[idx, "eco_vo"] = -robust_z(g[_eco_vo], g[_eco_vo])
            if g[_eco_cad].notna().sum() >= min_n: target.loc[idx, "eco_cad"] = -np.abs(robust_z(g[_eco_cad], g[_eco_cad]))
            if g[_eco_stride].notna().sum() >= min_n: target.loc[idx, "eco_stride"] = robust_z(g[_eco_stride], g[_eco_stride])

        for _, g in eco.groupby(group_cols, dropna=False):
            if len(g) >= CFG["res_min_group"]: _fill_eco_group(g, eco)
        still_nan = eco["eco_gct"].isna() & eco["eco_vr"].isna() & eco["eco_vo"].isna() & eco["eco_cad"].isna()
        if still_nan.any():
            for _, g in eco[still_nan].groupby(["speed_bin", "slope_bucket"], dropna=False):
                if len(g) >= CFG["res_min_group"]: _fill_eco_group(g, eco)

        rw = CFG["res_weights"]
        raw = (
            rw["hrpw"] * eco["eco_hrpw"].fillna(0) + rw["pps"] * eco["eco_pps"].fillna(0)
            + rw["gct"] * eco["eco_gct"].fillna(0) + rw["vr"] * eco["eco_vr"].fillna(0)
            + rw["vo"] * eco["eco_vo"].fillna(0) + rw["cad"] * eco["eco_cad"].fillna(0)
            + rw["stride"] * eco["eco_stride"].fillna(0)
        )
        p5, p95 = np.nanpercentile(raw, 5), np.nanpercentile(raw, 95)
        eco["RES_plus"] = (100 * (raw - p5) / (p95 - p5 + 1e-9)).clip(0, 100)
        df.loc[eco.index, "RES_plus"] = eco["RES_plus"].values

    # TSS proxy
    df["dur_sec"] = np.nan
    time_cands = [c for c in ["Idő", "Menetidő", "Eltelt idő"] if c in df.columns]
    if time_cands: df["dur_sec"] = df[time_cands[0]].apply(duration_to_seconds)
    df["TSS_proxy"] = compute_tss_proxy(df, hrmax=185)
    df = compute_asymmetry(df)
    return df


# =========================================================
# ADATFORRÁS: STRAVA + GARMIN CSV
# =========================================================
st.sidebar.header("Adatforrás")
strava_df, strava_source = render_strava_connect_sidebar()

st.sidebar.divider()
st.sidebar.header("📂 Garmin CSV / XLSX")
st.sidebar.caption(
    "Running Dynamics adatokhoz (GCT, VO, VR) szükséges. "
    "Strava-val kombinálva a teljes elemzés elérhető."
)
uploaded = st.sidebar.file_uploader("Garmin export (XLSX ajánlott)", type=["xlsx", "csv"])


def _merge_strava_garmin(strava_df: pd.DataFrame, garmin_df: pd.DataFrame) -> pd.DataFrame:
    if strava_df.empty: return garmin_df
    if garmin_df.empty: return strava_df
    s = strava_df.copy(); g = garmin_df.copy()
    for _df in (s, g):
        if "Dátum" in _df.columns:
            dt = pd.to_datetime(_df["Dátum"], errors="coerce")
            if dt.dt.tz is not None: dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
            _df["Dátum"] = dt
    s["_date"] = s["Dátum"].dt.date; g["_date"] = g["Dátum"].dt.date
    rd_cols = [c for c in ["vr_num", "gct_num", "vo_num", "stride_num",
                            "asym_gct_pct", "asym_stride_pct", "asym_power_pct", "Asymmetry_score"]
               if c in g.columns]
    if rd_cols:
        g_rd = g[["_date"] + rd_cols].copy()
        s = s.merge(g_rd, on="_date", how="left", suffixes=("", "_garmin"))
        for col in rd_cols:
            garmin_col = f"{col}_garmin"
            if garmin_col in s.columns:
                s[col] = s[col].combine_first(s[garmin_col]); s.drop(columns=[garmin_col], inplace=True)
    s.drop(columns=["_date"], inplace=True, errors="ignore")
    s_dates = set(s["Dátum"].dt.date)
    g_only = g[~g["_date"].isin(s_dates)].drop(columns=["_date"], errors="ignore")
    merged = pd.concat([s, g_only], ignore_index=True).sort_values("Dátum")
    return merged


if strava_df is None and uploaded is None:
    st.title("🏃 Garmin Futás Dashboard")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🟠 Automatikus szinkron")
        st.info("Csatlakozz Stravához a bal oldali sávban – minden futásod automatikusan betöltődik, CSV feltöltés nélkül.")
    with col_r:
        st.markdown("### 📂 Manuális feltöltés")
        st.info("Vagy tölts fel egy Garmin Connect XLSX exportot. Ez szükséges a teljes biomechanikai elemzéshez (GCT, VO, VR, aszimmetria).")
    st.stop()

with st.spinner("Adatok feldolgozása…"):
    if strava_df is not None and uploaded is not None:
        file_bytes = uploaded.getvalue()
        garmin_df = full_pipeline(file_bytes, uploaded.name, _v=3)
        garmin_dates = set(garmin_df["Dátum"].dt.date) if "Dátum" in garmin_df.columns else set()
        if "Dátum" in strava_df.columns:
            strava_only = strava_df[~strava_df["Dátum"].dt.date.isin(garmin_dates)].copy()
            if len(strava_only) > 0:
                strava_only["TSS_proxy"] = compute_tss_proxy(strava_only, hrmax=185)
                strava_only = compute_asymmetry(strava_only)
                df = pd.concat([garmin_df, strava_only], ignore_index=True).sort_values("Dátum")
            else:
                df = garmin_df.copy()
        else:
            df = garmin_df.copy()
        _data_source = "hybrid"
    elif strava_df is not None:
        df = strava_df.copy()
        df["TSS_proxy"] = compute_tss_proxy(df, hrmax=185)
        df = compute_asymmetry(df)
        _data_source = "strava"
    else:
        file_bytes = uploaded.getvalue()
        df = full_pipeline(file_bytes, uploaded.name, _v=3)
        _data_source = "garmin"

# =========================================================
# ALAP SZŰRÉS
# =========================================================
st.title("🏃 Garmin Futás Dashboard")

if _data_source == "strava":
    st.info("🟠 **Strava alapú elemzés** – Running Dynamics adatok (GCT, VO, VR) hiányoznak. Tölts fel Garmin XLSX exportot is a teljes elemzéshez.")
elif _data_source == "hybrid":
    st.success("✅ **Hibrid mód** – Strava automatikus szinkron + Garmin Running Dynamics. Minden elemzés elérhető.")

if "Dátum" not in df.columns:
    st.error("Nem találom a 'Dátum' oszlopot."); st.stop()

d = df[df["Dátum"].notna()].copy()
if d.empty:
    st.error("Nincs érvényes dátummal rendelkező sor."); st.stop()

run_type_col = find_col(d, "Run_type")
fatigue_col = find_col(d, "Fatigue_score")
fatigue_type_col = find_col(d, "Fatigue_type")
slope_col = find_col(d, "slope_bucket")
hr_col = find_col(d, "hr_num", "Átlagos pulzusszám")
pace_col = find_col(d, "Átlagos tempó")
pwr_col = find_col(d, "power_avg_w", "Átl. teljesítmény")

if run_type_col: d["Edzés típusa"] = d[run_type_col].apply(run_type_hu)
if slope_col: d["Terep"] = d[slope_col].apply(slope_bucket_hu)

# =========================================================
# SIDEBAR: HRmax + Szűrők + Baseline
# =========================================================
st.sidebar.divider()
st.sidebar.header("👤 Antropometriai adatok")
st.sidebar.caption("Súly, magasság és kor alapján pontosabb elemzés.")

for key, default in [("weight_kg", CFG["weight_kg_default"]), ("height_cm", CFG["height_cm_default"]), ("age", CFG["age_default"])]:
    if key not in st.session_state: st.session_state[key] = default

weight_kg = st.sidebar.number_input("Testsúly (kg)", min_value=40, max_value=150, value=int(st.session_state.weight_kg), step=1, key="weight_kg_input")
height_cm = st.sidebar.number_input("Magasság (cm)", min_value=140, max_value=220, value=int(st.session_state.height_cm), step=1, key="height_cm_input")
user_age  = st.sidebar.number_input("Kor (év)", min_value=15, max_value=85, value=int(st.session_state.age), step=1, key="age_input")
st.session_state.weight_kg = weight_kg; st.session_state.height_cm = height_cm; st.session_state.age = user_age

_hrmax_suggested = hrmax_from_age(user_age)
st.sidebar.caption(f"Javasolt HRmax (Tanaka-képlet, {user_age} év): **{_hrmax_suggested} bpm**")
st.sidebar.divider()
st.sidebar.header("Intenzitás (HR zónák)")
if "hrmax" not in st.session_state: st.session_state.hrmax = _hrmax_suggested
st.session_state.hrmax = st.sidebar.number_input(
    "HRmax (ütés/perc) – zónákhoz", min_value=120, max_value=240,
    value=int(st.session_state.hrmax), step=1,
    help=f"Tanaka-képlet szerint ({user_age} évhez): {_hrmax_suggested} bpm.",
)
hrmax = int(st.session_state.hrmax)
hr_zones = age_adjusted_hr_zones(hrmax, user_age)
fatmax_hr_pct_lo, fatmax_hr_pct_hi = fatmax_hr_pct_from_age(user_age)
recovery_age_factor = recovery_days_age_factor(user_age)

_bmi = bmi(weight_kg, height_cm)
_bmi_cat = "Sovány" if _bmi < 18.5 else "Normál" if _bmi < 25.0 else "Túlsúlyos" if _bmi < 30.0 else "Elhízott"
st.sidebar.caption(
    f"BMI: **{_bmi:.1f}** ({_bmi_cat}) | "
    f"Fatmax HR: **{round(hrmax * fatmax_hr_pct_lo)}–{round(hrmax * fatmax_hr_pct_hi)} bpm** | "
    f"Recovery szorzó: **×{recovery_age_factor:.1f}**"
)

st.sidebar.header("Szűrők")
min_date = pd.to_datetime(d["Dátum"], errors="coerce").dropna().min()
max_date = pd.to_datetime(d["Dátum"], errors="coerce").dropna().max()
if pd.isna(min_date) or pd.isna(max_date):
    st.sidebar.error("Nem sikerült dátumtartományt képezni."); st.stop()

dt_input = st.sidebar.date_input("Dátumtartomány", value=(min_date.date(), max_date.date()))
date_from, date_to = (dt_input[0], dt_input[1]) if len(dt_input) == 2 else (dt_input[0], dt_input[0])
mask = (d["Dátum"].dt.date >= date_from) & (d["Dátum"].dt.date <= date_to)

if run_type_col:
    types = ["easy", "tempo", "race", "interval"]
    present = [t for t in types if t in set(d[run_type_col].dropna().astype(str))]
    selected_types = st.sidebar.multiselect(
        "Edzés típusa", options=present, default=present,
        format_func=lambda x: {"easy": "Könnyű", "tempo": "Tempó", "race": "Verseny", "interval": "Interval"}.get(x, x),
    )
    if selected_types: mask &= d[run_type_col].isin(selected_types)

if slope_col:
    slope_opts = d[slope_col].dropna().unique().tolist()
    slope_sel = st.sidebar.multiselect("Terep", options=slope_opts, default=slope_opts, format_func=slope_bucket_hu)
    if slope_sel: mask &= d[slope_col].isin(slope_sel)

if fatigue_col and d[fatigue_col].notna().sum() > 0:
    fmin = float(np.nanmin(d[fatigue_col])); fmax = float(np.nanmax(d[fatigue_col]))
    sel_fmin, sel_fmax = st.sidebar.slider("Fatigue_score", min_value=fmin, max_value=fmax, value=(fmin, fmax))
    mask &= d[fatigue_col].between(sel_fmin, sel_fmax)

st.sidebar.divider()
st.sidebar.header("Baseline (B-line)")
baseline_mode = st.sidebar.selectbox("Baseline mód", options=["Auto (Edzés típusa szerint)", "Mindig EASY baseline"], index=0)
baseline_weeks = st.sidebar.slider("Baseline ablak (hetek)", min_value=4, max_value=52, value=CFG["baseline_weeks_default"], step=1)
baseline_min_runs = st.sidebar.slider("Minimum baseline futás", min_value=10, max_value=80, value=CFG["baseline_min_runs_default"], step=5)

st.sidebar.divider()
st.sidebar.header("Export")
view_for_export = d.loc[mask].copy().sort_values("Dátum")
csv_export = view_for_export.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(label="⬇️ Szűrt adatok letöltése (CSV)", data=csv_export, file_name="garmin_filtered.csv", mime="text/csv")

st.sidebar.divider()
if st.sidebar.button("Kijelentkezés"):
    st.session_state.auth_ok = False; st.rerun()

view = d.loc[mask].copy().sort_values("Dátum")

# =========================================================
# TABOK
# =========================================================
tab_overview, tab_last, tab_warn, tab_ready, tab_pmc, tab_recovery, tab_asym, tab_race, tab_heat, tab_strava, tab_strava_analysis, tab_ai, tab_data = st.tabs(
    ["📌 Áttekintés", "🔎 Utolsó futás", "🚦 Warning", "🏁 Readiness",
     "📈 PMC & Kockázat", "🔄 Recovery", "⚖️ Aszimmetria",
     "🏆 Versenyidő", "🌡️ Hőkorrekció",
     "🟠 Strava adatok", "🟠 Strava elemzés", "🤖 AI Edző", "📄 Adatok"]
)

# =========================================================
# TAB: ÁTTEKINTÉS
# =========================================================
with tab_overview:
    st.subheader("🏃 Easy run target – mai állapot")
    _orig_lo = CFG["easy_target_hr_pct_lo"]; _orig_hi = CFG["easy_target_hr_pct_hi"]
    CFG["easy_target_hr_pct_lo"] = fatmax_hr_pct_lo; CFG["easy_target_hr_pct_hi"] = fatmax_hr_pct_hi
    _et = compute_easy_target(
        df=d, hrmax=hrmax, fatigue_col=fatigue_col, run_type_col=run_type_col,
        hr_col=hr_col, pace_col=pace_col, pwr_col=pwr_col,
        baseline_weeks=baseline_weeks, baseline_min_runs=baseline_min_runs,
    )
    CFG["easy_target_hr_pct_lo"] = _orig_lo; CFG["easy_target_hr_pct_hi"] = _orig_hi

    if _et is None:
        st.info("Easy run target-hez legalább 5 easy futás szükséges HR adattal.")
    else:
        rs = _et["readiness_status"]; rt = _et["readiness_text"]
        if rs == "🔴": st.error(f"{rs} {rt}")
        elif rs == "🟠": st.warning(f"{rs} {rt}")
        elif rs == "🟡": st.warning(f"{rs} {rt}")
        else: st.success(f"{rs} {rt}")

        st.markdown("#### 🎯 Mai easy zóna")
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            st.markdown(f'<div style="background:#1a1a2e;border-radius:14px;padding:18px 20px;text-align:center;"><div style="font-size:0.85rem;color:#aaa;margin-bottom:4px;">❤️ Pulzus</div><div style="font-size:2rem;font-weight:700;color:#e74c3c;">{_et["hr_lo"]}–{_et["hr_hi"]}</div><div style="font-size:0.9rem;color:#aaa;">bpm</div></div>', unsafe_allow_html=True)
        with _c2:
            st.markdown(f'<div style="background:#1a1a2e;border-radius:14px;padding:18px 20px;text-align:center;"><div style="font-size:0.85rem;color:#aaa;margin-bottom:4px;">⏱️ Tempó</div><div style="font-size:2rem;font-weight:700;color:#3498db;">{_et["pace_lo"]}–{_et["pace_hi"]}</div><div style="font-size:0.9rem;color:#aaa;">min/km</div></div>', unsafe_allow_html=True)
        with _c3:
            _pwr_str = f"{int(_et['pwr_lo'])}–{int(_et['pwr_hi'])}" if (pd.notna(_et["pwr_lo"]) and pd.notna(_et["pwr_hi"])) else "—"
            _pwr_unit = "W" if _pwr_str != "—" else "nincs power adat"
            st.markdown(f'<div style="background:#1a1a2e;border-radius:14px;padding:18px 20px;text-align:center;"><div style="font-size:0.85rem;color:#aaa;margin-bottom:4px;">⚡ Teljesítmény</div><div style="font-size:2rem;font-weight:700;color:#2ecc71;">{_pwr_str}</div><div style="font-size:0.9rem;color:#aaa;">{_pwr_unit}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("ℹ️ Számítás részletei"):
            if pd.notna(_et["fatmax_hr"]): st.markdown(f"- Fatmax HR: **{_et['fatmax_hr']:.0f} bpm** ({_et['fatmax_pace']} min/km)")
            if pd.notna(_et["hr_med_easy"]): st.markdown(f"- Easy medián HR: **{_et['hr_med_easy']:.0f} bpm**")
            if pd.notna(_et["pace_med_easy"]) and _et["pace_med_easy"] != "—": st.markdown(f"- Easy medián tempó: **{_et['pace_med_easy']} min/km**")
            if pd.notna(_et["current_fatigue"]): st.markdown(f"- Jelenlegi Fatigue: **{_et['current_fatigue']:.0f}**")

        if not _et["easy_history"].empty and hr_col and hr_col in _et["easy_history"].columns:
            st.markdown("#### 📊 Legutóbbi easy futások HR vs. mai célsáv")
            _hist = _et["easy_history"].copy()
            _hist["HR"] = pd.to_numeric(_hist[hr_col], errors="coerce")
            _hist = _hist.dropna(subset=["Dátum", "HR"])
            if len(_hist) >= 3:
                fig_et = px.scatter(_hist, x="Dátum", y="HR", labels={"Dátum": "Dátum", "HR": "Átlag HR (bpm)"}, title="Easy futások átlag HR-je – célsávhoz viszonyítva", color_discrete_sequence=["#3498db"], opacity=0.75)
                fig_et.add_hrect(y0=_et["hr_lo"], y1=_et["hr_hi"], fillcolor="green", opacity=0.12, annotation_text=f"Mai célsáv ({_et['hr_lo']}–{_et['hr_hi']} bpm)", annotation_position="top left")
                st.plotly_chart(fig_et, use_container_width=True)

    st.divider()
    st.subheader("🗓️ Napi összkép (coach)")
    status, msg = daily_coach_summary(base_all=d, run_type_col=run_type_col, fatigue_col=fatigue_col, baseline_weeks=baseline_weeks, baseline_min_runs=baseline_min_runs)
    if status == "🔴": st.error(f"{status} {msg}")
    elif status == "🟠": st.warning(f"{status} {msg}")
    else: st.success(f"{status} {msg}")

    st.divider()
    tech_avg = view["Technika_index"].mean() if ("Technika_index" in view.columns and view["Technika_index"].notna().any()) else np.nan
    fat_avg = view[fatigue_col].mean() if (fatigue_col and view[fatigue_col].notna().any()) else np.nan
    most_type = run_type_hu(view[run_type_col].value_counts().index[0]) if (run_type_col and len(view) > 0) else "—"
    _pwr_avg = view["power_avg_w"].mean() if "power_avg_w" in view.columns and view["power_avg_w"].notna().any() else np.nan
    _wkg_avg = power_per_kg(_pwr_avg, weight_kg) if pd.notna(_pwr_avg) else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Futások (szűrve)", f"{len(view)}")
    c2.metric("Átlag Technika_index", f"{tech_avg:.1f}" if pd.notna(tech_avg) else "—")
    c3.metric("Átlag Fatigue_score", f"{fat_avg:.1f}" if pd.notna(fat_avg) else "—")
    c4.metric("Átlag W/kg", f"{_wkg_avg:.2f}" if pd.notna(_wkg_avg) else "—", delta=f"{weight_kg} kg alapon")
    c5.metric("Leggyakoribb típus", most_type)

    st.divider()
    st.subheader("📊 Heti terhelés & ramp rate")
    base_all = d[d["Dátum"].notna()].sort_values("Dátum").copy()
    time_col_candidates = [c for c in ["Idő", "Menetidő", "Eltelt idő"] if c in base_all.columns]
    if time_col_candidates: base_all["dur_sec"] = base_all[time_col_candidates[0]].apply(duration_to_seconds)
    else: base_all["dur_sec"] = np.nan

    weekly = (
        base_all.set_index("Dátum").resample("W-MON")
        .agg(week_km=("dist_km", "sum"), week_sec=("dur_sec", "sum"), week_elev=("asc_m", "sum"))
        .reset_index().rename(columns={"Dátum": "week"}).sort_values("week")
    )
    weekly["week_hours"] = weekly["week_sec"] / 3600.0

    rr_km = rr_time = rr_elev = np.nan
    if len(weekly) >= 6:
        last_idx = len(weekly) - 1
        if weekly.loc[last_idx, "week_km"] < 0.3 * max(1e-9, weekly.loc[last_idx - 1, "week_km"]): last_idx -= 1
        last_week = weekly.loc[last_idx]; prev4 = weekly.loc[last_idx - 4: last_idx - 1]
        def ramp(curr, prev_mean):
            if pd.isna(curr) or pd.isna(prev_mean) or prev_mean <= 0: return np.nan
            return (curr - prev_mean) / prev_mean * 100.0
        rr_km = ramp(float(last_week["week_km"]), float(np.nanmean(prev4["week_km"])))
        rr_time = ramp(float(last_week["week_hours"]), float(np.nanmean(prev4["week_hours"])))
        rr_elev = ramp(float(last_week["week_elev"]), float(np.nanmean(prev4["week_elev"])))

    metric = st.selectbox("Melyik terhelést nézzük?", ["Heti km", "Heti idő (óra)", "Heti emelkedés (m)"], key="load_metric")
    y_col = {"Heti km": "week_km", "Heti idő (óra)": "week_hours", "Heti emelkedés (m)": "week_elev"}[metric]
    rr_val = {"Heti km": rr_km, "Heti idő (óra)": rr_time, "Heti emelkedés (m)": rr_elev}[metric]
    ylabel = {"Heti km": "km", "Heti idő (óra)": "óra", "Heti emelkedés (m)": "m"}[metric]

    def ramp_badge(rr):
        if pd.isna(rr): return ("⚪", "Ramp rate: nincs elég adat")
        if rr <= CFG["ramp_warn"]: return ("🟢", f"Ramp rate: {rr:+.1f}% (biztonságos)")
        if rr <= CFG["ramp_red"]: return ("🟠", f"Ramp rate: {rr:+.1f}% (figyelmeztető)")
        return ("🔴", f"Ramp rate: {rr:+.1f}% (túl gyors emelés)")

    badge, badge_txt = ramp_badge(rr_val)
    st.caption(f"{badge} {badge_txt}")
    st.plotly_chart(px.bar(weekly, x="week", y=y_col, title=f"Heti terhelés – {metric}", labels={"week": "Hét", y_col: ylabel}), use_container_width=True)

    daily = base_all.copy()
    daily["date"] = daily["Dátum"].dt.date
    daily = daily.groupby("date", as_index=False).agg(km=("dist_km", "sum"), sec=("dur_sec", "sum"), elev=("asc_m", "sum"))
    daily["date"] = pd.to_datetime(daily["date"]); daily = daily.sort_values("date")
    daily["km_7"] = daily["km"].rolling(7, min_periods=3).sum()
    daily["km_28"] = daily["km"].rolling(28, min_periods=10).sum()
    daily["h_7"] = daily["sec"].rolling(7, min_periods=3).sum() / 3600.0
    daily["h_28"] = daily["sec"].rolling(28, min_periods=10).sum() / 3600.0
    daily["elev_7"] = daily["elev"].rolling(7, min_periods=3).sum()
    daily["elev_28"] = daily["elev"].rolling(28, min_periods=10).sum()
    roll_cols = {"Heti km": ["km_7", "km_28"], "Heti idő (óra)": ["h_7", "h_28"], "Heti emelkedés (m)": ["elev_7", "elev_28"]}[metric]
    st.plotly_chart(px.line(daily, x="date", y=roll_cols, title="Gördülő összeg – 7 nap vs 28 nap"), use_container_width=True)

    with st.expander("📋 Heti táblázat"):
        st.dataframe(weekly.tail(24), use_container_width=True, hide_index=True)

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
                heatmap_data, labels=dict(x="Hét", y="Nap", color="km"),
                x=heatmap_data.columns,
                y=["Hétfő", "Kedd", "Szerda", "Csütörtök", "Péntek", "Szombat", "Vasárnap"],
                color_continuous_scale="Greens", aspect="auto", title=f"{sel_year} – Futott km naponta",
            )
            fig_cal.update_xaxes(side="top", dtick=2)
            fig_cal.update_layout(height=300, margin=dict(l=0, r=0, t=50, b=0))
            fig_cal.update_traces(xgap=2, ygap=2)
            st.plotly_chart(fig_cal, use_container_width=True)

    st.divider()
    st.subheader("📈 Technika_index időben")
    if "Technika_index" in view.columns and view["Technika_index"].notna().sum() >= 3:
        fig = px.scatter(
            _safe_dropna(view, ["Technika_index"]), x="Dátum", y="Technika_index",
            color="Edzés típusa" if "Edzés típusa" in view.columns else None,
            symbol="Terep" if "Terep" in view.columns else None,
            hover_data=[c for c in ["Cím", "Átlagos tempó", fatigue_col, fatigue_type_col, "Terep"] if c and c in view.columns],
            opacity=0.75,
        )
        view2 = view[["Dátum", "Technika_index"]].dropna().sort_values("Dátum").copy()
        if len(view2) >= 10:
            view2["roll30"] = view2["Technika_index"].rolling(window=30, min_periods=10).mean()
            for tr in px.line(view2, x="Dátum", y="roll30").data: fig.add_trace(tr)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nincs elég Technika_index adat.")

# =========================================================
# TAB: UTOLSÓ FUTÁS
# =========================================================
with tab_last:
    st.subheader("🔎 Utolsó futás elemzése")
    base_all_runs = d.dropna(subset=["Dátum"]).sort_values("Dátum").copy()
    base = _safe_dropna(d, ["Dátum", "Technika_index"]).sort_values("Dátum")

    if len(base_all_runs) == 0:
        st.info("Nincs futás adat.")
    else:
        options = base_all_runs.tail(60).copy()
        def label_row(r):
            title = r.get("Cím", "") if "Cím" in options.columns else ""
            rt = run_type_hu(r.get("Run_type", "")) if "Run_type" in options.columns else ""
            has_tech = "✓" if pd.notna(r.get("Technika_index")) else "—"
            return f"{r['Dátum'].strftime('%Y-%m-%d %H:%M')} | {rt} | {title} [{has_tech}]"[:120]
        options["__label"] = options.apply(label_row, axis=1)
        chosen_label = st.selectbox("Futás kiválasztása (✓ = van Technika_index)", options["__label"].tolist(), index=len(options)-1, key="pick_last")
        last = options.loc[options["__label"] == chosen_label].iloc[0]

        last_type = None
        if run_type_col and run_type_col in base.columns and pd.notna(last.get(run_type_col)):
            last_type = str(last.get(run_type_col)).strip().lower()

        if baseline_mode == "Auto (Edzés típusa szerint)" and last_type in ("easy", "tempo", "race", "interval"):
            baseline_full = get_type_baseline(base_df=base, last_date=last["Dátum"], run_type_col=run_type_col, target_type=last_type, weeks=baseline_weeks, min_runs=baseline_min_runs)
        else:
            baseline_full = get_easy_baseline(base_df=base, last_date=last["Dátum"], weeks=baseline_weeks, min_runs=baseline_min_runs)

        st.divider()
        st.subheader("🔥 Fatmax & Aerobic decoupling")
        hr_col_l = find_col(base, "Átlagos pulzusszám", "Átlagos pulzusszám ")
        pace_col_f = find_col(base, "Átlagos tempó")
        pwr_col_l = find_col(base, "Átl. teljesítmény", "Átlagos teljesítmény", "Average Power", "Avg Power")
        pwr_max_col = find_col(base, "Max. teljesítmény", "Maximum Power", "Max Power")

        if hr_col_l is None or pace_col_f is None:
            st.info("Fatmax-hoz kell: **Átlagos pulzusszám** és **Átlagos tempó** oszlop.")
        else:
            fat_base = baseline_full.copy() if baseline_full is not None and len(baseline_full) > 0 else base.copy()
            fatmax = estimate_fatmax_from_runs(fat_base, hrmax, hr_col_l, pace_col_f, pwr_col_l, min_points=12)
            dec = aerobic_decoupling_proxy(last, baseline_full, hr_col_l, pace_col_f, pwr_col_l)
            k1, k2, k3, k4 = st.columns(4)
            if fatmax is None:
                k1.metric("Fatmax HR", "—"); k2.metric("Fatmax tempó", "—"); k3.metric("Fatmax power", "—"); k4.metric("Decoupling", "—")
                st.caption("Kevés adat a Fatmax becsléshez.")
            else:
                k1.metric("Fatmax HR", f"{fatmax['hr_fatmax']:.0f} bpm")
                k2.metric("Fatmax tempó", f"{fatmax['pace_fatmax']}/km")
                k3.metric("Fatmax power", f"{fatmax['pwr_fatmax']:.0f} W" if pd.notna(fatmax["pwr_fatmax"]) else "—")
                k4.metric("Aerobic decoupling", f"{dec['delta_pct']:+.1f}%" if dec else "—")

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Technika_index", f"{float(last['Technika_index']):.1f}" if pd.notna(last.get("Technika_index")) else "—")
        fat_val = last.get("Fatigue_score")
        c2.metric("Fatigue_score", f"{float(fat_val):.1f}" if pd.notna(fat_val) else "—")
        c3.metric("Edzés típusa", run_type_hu(last.get("Run_type")) if pd.notna(last.get("Run_type")) else "—")
        pace_v = last.get("Átlagos tempó") if "Átlagos tempó" in base.columns else None
        dist_v = last.get("Távolság") if "Távolság" in base.columns else None
        c4.metric("Tempó / Táv", f"{pace_v} / {dist_v} km" if (pace_v or dist_v) else "—")

        if len(baseline_full) >= 8:
            compare_cols = [
                ("Átl. pedálütem", "Cadence (spm)", "cadence_stability"),
                ("Átlagos lépéshossz", "Lépéshossz (m)", "higher_better"),
                ("Átlagos függőleges arány", "Vertical Ratio (%)", "lower_better"),
                ("Átlagos függőleges oszcilláció", "Vertical Osc (cm)", "lower_better"),
                ("Átlagos talajérintési idő", "GCT (ms)", "lower_better"),
                ("Átlagos pulzusszám", "Átlag pulzus", "context"),
            ]
            rows = []
            for col_c, label_c, rule_c in compare_cols:
                if col_c not in base.columns: continue
                v = num(last.get(col_c)); b_v = med(baseline_full[col_c])
                if pd.isna(v) or pd.isna(b_v) or b_v == 0: continue
                delta_pct = (v - b_v) / b_v * 100.0
                good = (-delta_pct if rule_c == "lower_better" else delta_pct if rule_c == "higher_better" else -abs(delta_pct) if rule_c == "cadence_stability" else 0.0)
                rows.append([label_c, float(v), float(b_v), float(delta_pct), float(good), rule_c])
            if rows:
                comp = pd.DataFrame(rows, columns=["Mutató", "Utolsó", "Baseline", "Eltérés_%", "Jó_irány", "rule"])
                st.markdown("### 📈 Eltérések a baseline-hoz képest")
                st.plotly_chart(px.bar(comp.sort_values("Jó_irány"), x="Jó_irány", y="Mutató", orientation="h", hover_data=["Utolsó", "Baseline", "Eltérés_%"]), use_container_width=True)
                for _, r in comp.iterrows():
                    if r["rule"] in ("lower_better", "higher_better"):
                        if r["Jó_irány"] < -5: st.write(f"🔴 **{r['Mutató']}** romlott (≈ {r['Eltérés_%']:+.1f}%)")
                        elif r["Jó_irány"] < -2: st.write(f"🟠 **{r['Mutató']}** kicsit romlott (≈ {r['Eltérés_%']:+.1f}%)")
                        else: st.write(f"🟢 **{r['Mutató']}** rendben (≈ {r['Eltérés_%']:+.1f}%)")

# =========================================================
# TAB: WARNING
# =========================================================
with tab_warn:
    st.subheader("🚦 Warning rendszer")
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

        warn_base = _safe_dropna(d, ["Dátum", "Technika_index"]).sort_values("Dátum")
        easy_w = (warn_base[warn_base[run_type_col] == "easy"].copy() if run_type_col else warn_base.copy())
        if len(easy_w) > 0:
            last_day_w = easy_w["Dátum"].max(); start_w = last_day_w - pd.Timedelta(weeks=baseline_weeks)
            easy_w = easy_w[easy_w["Dátum"] >= start_w].copy()
            if len(easy_w) < baseline_min_runs: easy_w = easy_w.tail(baseline_min_runs).copy()
        easy_f = easy_w.dropna(subset=[fatigue_col]).copy()
        if len(easy_f) < 5:
            st.info("Nincs elég easy + Fatigue_score adat.")
        else:
            last_red_df = easy_f.tail(n_red).copy(); last_yellow_df = easy_f.tail(n_yellow).copy()
            last_red_df["hit_red"] = (last_red_df["Technika_index"] < tech_red) & (last_red_df[fatigue_col] > fat_red)
            last_yellow_df["hit_yellow"] = (last_yellow_df["Technika_index"] < tech_yellow) | (last_yellow_df[fatigue_col] > fat_yellow)
            red_hits = int(last_red_df["hit_red"].sum()); yellow_hits = int(last_yellow_df["hit_yellow"].sum())
            if red_hits >= need_red: st.error(f"🔴 PIROS — Utolsó {n_red} easy futásból {red_hits} találat: Tech < {tech_red} ÉS Fatigue > {fat_red}.")
            elif yellow_hits >= need_yellow: st.warning(f"🟠 SÁRGA — Utolsó {n_yellow} easy futásból {yellow_hits} találat.")
            else: st.success("🟢 ZÖLD — Stabil easy technika / fáradás kontrollált.")
            st.metric("PIROS találatok", f"{red_hits}/{n_red}"); st.metric("SÁRGA találatok", f"{yellow_hits}/{n_yellow}")

# =========================================================
# TAB: READINESS
# =========================================================
with tab_ready:
    st.subheader("🏁 Verseny-előrejelzés (Readiness) – 14 napos ablak")
    if run_type_col is None or fatigue_col is None or "Technika_index" not in d.columns:
        st.info("Readiness-hez kell Edzés típusa + Fatigue_score + Technika_index.")
    else:
        ready_base = _safe_dropna(d, ["Dátum", "Technika_index"]).sort_values("Dátum")
        easy_r = ready_base[ready_base[run_type_col] == "easy"].dropna(subset=[fatigue_col]).copy()
        if len(easy_r) < 10:
            st.info("Nincs elég easy + Fatigue_score adat.")
        else:
            default_date = ready_base["Dátum"].max().date()
            race_date = st.date_input("Verseny dátuma", value=default_date, key="race_date")
            window_days = st.slider("Ablak (nap)", 7, 28, 14, key="race_window")
            start_r = pd.Timestamp(race_date) - pd.Timedelta(days=window_days)
            w_r = easy_r[(easy_r["Dátum"] >= start_r) & (easy_r["Dátum"] <= pd.Timestamp(race_date))].copy()
            if len(w_r) < 3:
                st.warning(f"Az utolsó {window_days} napban kevés easy futás van ({len(w_r)}).")
            else:
                tech_mean_r = float(w_r["Technika_index"].mean()); fat_mean_r = float(w_r[fatigue_col].mean())
                w2_r = w_r.sort_values("Dátum")[["Dátum", "Technika_index"]].dropna().copy()
                x_r = (w2_r["Dátum"] - w2_r["Dátum"].min()).dt.total_seconds().to_numpy()
                y_r = w2_r["Technika_index"].to_numpy()
                slope_per_day = float(np.polyfit(x_r, y_r, 1)[0]) * 86400.0 if len(w2_r) >= 3 and np.nanstd(x_r) > 0 else 0.0
                red_hits_r = int(((w_r["Technika_index"] < 35) & (w_r[fatigue_col] > 55)).sum())
                tech_p25, tech_p75 = np.nanpercentile(easy_r["Technika_index"], [25, 75])
                fat_p25, fat_p75 = np.nanpercentile(easy_r[fatigue_col], [25, 75])
                tech_score_r = float(np.clip(100 * (tech_mean_r - tech_p25) / (tech_p75 - tech_p25 + 1e-9), 0, 100))
                fat_score_r  = float(np.clip(100 * (fat_p75 - fat_mean_r) / (fat_p75 - fat_p25 + 1e-9), 0, 100))
                trend_score_r = float(np.clip(50 + 20 * slope_per_day, 0, 100))
                readiness = float(np.clip(0.50 * tech_score_r + 0.30 * fat_score_r + 0.20 * trend_score_r - min(30, red_hits_r * 10), 0, 100))
                c1_r, c2_r, c3_r, c4_r = st.columns(4)
                c1_r.metric("Readiness_score", f"{readiness:.0f}"); c2_r.metric("Tech átlag", f"{tech_mean_r:.1f}")
                c3_r.metric("Fatigue átlag", f"{fat_mean_r:.1f}"); c4_r.metric("Tech trend", f"{slope_per_day:+.2f}")
                if readiness >= 60 and red_hits_r == 0: st.success("🟢 Jó verseny-készenlét.")
                elif readiness < 40 or red_hits_r >= 2: st.error("🔴 Nem ideális (fáradtság magas / technika instabil).")
                else: st.warning("🟠 Közepes készenlét.")
                st.plotly_chart(px.line(w_r.sort_values("Dátum"), x="Dátum", y=["Technika_index", fatigue_col]), use_container_width=True)

# =========================================================
# TAB: PMC & SÉRÜLÉSKOCKÁZAT
# =========================================================
with tab_pmc:
    st.subheader("📈 Performance Management Chart – CTL / ATL / TSB")
    d_pmc = d.copy()
    if "TSS_proxy" in d_pmc.columns: d_pmc["TSS_proxy"] = compute_tss_proxy(d_pmc, hrmax)
    acwr_df = compute_acwr(d_pmc, load_col="TSS_proxy", acute_days=CFG["acwr_acute_days"], chronic_days=CFG["acwr_chronic_days"])
    if acwr_df.empty:
        st.info("Nincs elég TSS adat a PMC-hez.")
    else:
        pmc_df = compute_ctl_atl_tsb(acwr_df)
        last_pmc = pmc_df.iloc[-1]
        acwr_now = float(last_pmc.get("acwr", np.nan)); tsb_now = float(last_pmc.get("TSB", np.nan))
        ctl_now = float(last_pmc.get("CTL", np.nan)); atl_now = float(last_pmc.get("ATL", np.nan))
        rr_pmc = np.nan
        if len(acwr_df) >= 35:
            last7 = acwr_df.tail(7)["load"].sum(); prev28_mean = acwr_df.tail(35).head(28)["load"].mean()
            if prev28_mean > 0: rr_pmc = (last7 / 7 - prev28_mean) / prev28_mean * 100
        fat_now = float(d_pmc.dropna(subset=[fatigue_col]).sort_values("Dátum").iloc[-1][fatigue_col]) if (fatigue_col and d_pmc[fatigue_col].notna().any()) else None
        asym_now = float(d_pmc.dropna(subset=["Asymmetry_score"]).sort_values("Dátum").iloc[-1]["Asymmetry_score"]) if ("Asymmetry_score" in d_pmc.columns and d_pmc["Asymmetry_score"].notna().any()) else None
        inj_score, inj_status, inj_expl = injury_risk_score(acwr_now, tsb_now, fat_now, asym_now, rr_pmc)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CTL (fittség)", f"{ctl_now:.1f}" if pd.notna(ctl_now) else "—")
        c2.metric("ATL (fáradtság)", f"{atl_now:.1f}" if pd.notna(atl_now) else "—")
        c3.metric("TSB (forma)", f"{tsb_now:+.1f}" if pd.notna(tsb_now) else "—", delta=last_pmc.get("TSB_status", ""))
        c4.metric("ACWR", f"{acwr_now:.2f}" if pd.notna(acwr_now) else "—", delta=last_pmc.get("acwr_status", ""))
        c5.metric("Sérüléskockázat", f"{inj_score:.0f}/100" if pd.notna(inj_score) else "—", delta=inj_status)
        if inj_status.startswith("🔴"): st.error(f"**{inj_status}** sérüléskockázat")
        elif inj_status.startswith("🟠"): st.warning(f"**{inj_status}** sérüléskockázat")
        else: st.success(f"**{inj_status}** sérüléskockázat")
        with st.expander("🔍 Kockázati komponensek", expanded=True):
            for e in inj_expl: st.write(f"• {e}")
        pmc_plot = pmc_df.tail(180).copy()
        st.plotly_chart(px.line(pmc_plot, x="date", y=["CTL", "ATL"], title="Fittség (CTL) vs Fáradtság (ATL)", color_discrete_map={"CTL": "#2ecc71", "ATL": "#e74c3c"}), use_container_width=True)
        fig_tsb = px.area(pmc_plot, x="date", y="TSB", title="Forma (TSB = CTL − ATL)", color_discrete_sequence=["#3498db"])
        fig_tsb.add_hline(y=CFG["tsb_green"], line_dash="dot", line_color="green", annotation_text="Forma zóna")
        fig_tsb.add_hline(y=CFG["tsb_red"], line_dash="dot", line_color="red", annotation_text="Kritikus")
        st.plotly_chart(fig_tsb, use_container_width=True)
        acwr_plot = acwr_df.tail(120).copy()
        fig_acwr = px.line(acwr_plot, x="date", y="acwr", title="ACWR idősor", color_discrete_sequence=["#e67e22"])
        fig_acwr.add_hrect(y0=CFG["acwr_safe_lo"], y1=CFG["acwr_safe_hi"], fillcolor="green", opacity=0.08, annotation_text="Optimális")
        fig_acwr.add_hrect(y0=CFG["acwr_safe_hi"], y1=CFG["acwr_warn_hi"], fillcolor="orange", opacity=0.08, annotation_text="Figyelmeztető")
        st.plotly_chart(fig_acwr, use_container_width=True)

# =========================================================
# TAB: RECOVERY
# =========================================================
with tab_recovery:
    st.subheader("🔄 Recovery Time Modell")
    if fatigue_col is None or "Technika_index" not in d.columns:
        st.info("Recovery modellhez kell Fatigue_score és Technika_index.")
    else:
        rec = compute_recovery_model(d, fatigue_col=fatigue_col, run_type_col=run_type_col, lookback_weeks=CFG["recovery_lookback_weeks"], min_events=CFG["recovery_min_events"])
        if rec is None:
            st.info("Nincs elég adat a recovery modellhez.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Medián recovery", f"{rec['median_recovery']:.0f} nap")
            c2.metric("Elemzett események", f"{rec['n_events']}")
            c3.metric("Jelenlegi Fatigue", f"{rec['current_fat']:.0f}" if rec["current_fat"] else "—")
            _pred_raw = rec["predicted_days"]
            _pred_age = round(_pred_raw * recovery_age_factor) if _pred_raw is not None else None
            c4.metric("Becsült recovery", f"{_pred_age} nap" if _pred_age is not None else "—", delta=f"×{recovery_age_factor:.1f} ({user_age} év)")
            if _pred_age is not None:
                if _pred_age <= 1: st.success(f"🟢 Valószínűleg felépültél – becsült recovery: {_pred_age} nap.")
                elif _pred_age <= 3: st.warning(f"🟠 Még {_pred_age} nap ajánlott könnyű edzés.")
                else: st.error(f"🔴 Becsült recovery: {_pred_age} nap – ne emelj terhelést még.")
            fig_rec = px.scatter(rec["df"], x="fatigue", y="recovery_days", labels={"fatigue": "Fatigue_score", "recovery_days": "Recovery napok"}, title="Fatigue_score vs Recovery idő")
            if len(rec["df"]) >= 3:
                _xf = rec["df"]["fatigue"].to_numpy(dtype=float); _yf = rec["df"]["recovery_days"].to_numpy(dtype=float)
                _m, _b = np.polyfit(_xf, _yf, 1); _xs = np.linspace(_xf.min(), _xf.max(), 60)
                fig_rec.add_scatter(x=_xs, y=_m*_xs+_b, mode="lines", name="Trend", line=dict(color="orange", dash="dash"))
            st.plotly_chart(fig_rec, use_container_width=True)

# =========================================================
# TAB: ASZIMMETRIA
# =========================================================
with tab_asym:
    st.subheader("⚖️ Futási aszimmetria elemzés")
    asym_available = any(c in d.columns and d[c].notna().any() for c in ["asym_gct_pct", "asym_stride_pct", "asym_power_pct", "Asymmetry_score"])
    if not asym_available:
        st.info("Nincs bal/jobb aszimmetria adat. Ehhez Garmin Running Dynamics szükséges.")
    else:
        asym_base = d.dropna(subset=["Dátum"]).sort_values("Dátum").copy()
        last_asym = asym_base.dropna(subset=["Asymmetry_score"]).iloc[-1] if asym_base["Asymmetry_score"].notna().any() else None
        if last_asym is not None:
            asym_val = float(last_asym["Asymmetry_score"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Összesített aszimmetria", f"{asym_val:.1f}%")
            c2.metric("GCT aszimmetria", f"{float(last_asym.get('asym_gct_pct', np.nan)):.1f}%" if pd.notna(last_asym.get("asym_gct_pct")) else "—")
            c3.metric("Lépéshossz aszimmetria", f"{float(last_asym.get('asym_stride_pct', np.nan)):.1f}%" if pd.notna(last_asym.get("asym_stride_pct")) else "—")
            c4.metric("Power balance", f"{float(last_asym.get('asym_power_pct', np.nan)):.1f}%" if pd.notna(last_asym.get("asym_power_pct")) else "—")
            if asym_val >= CFG["asym_red_pct"]: st.error(f"🔴 Magas aszimmetria ({asym_val:.1f}%) – sérülésprediktív jel.")
            elif asym_val >= CFG["asym_warn_pct"]: st.warning(f"🟠 Figyelemre méltó aszimmetria ({asym_val:.1f}%).")
            else: st.success(f"🟢 Jó szimmetria ({asym_val:.1f}%).")
        asym_cols_present = [c for c in ["asym_gct_pct", "asym_stride_pct", "asym_power_pct", "Asymmetry_score"] if c in asym_base.columns and asym_base[c].notna().any()]
        if asym_cols_present:
            asym_labels = {"asym_gct_pct": "GCT aszimmetria (%)", "asym_stride_pct": "Lépéshossz aszimmetria (%)", "asym_power_pct": "Power balance (%)", "Asymmetry_score": "Összesített (%)"}
            sel_asym = st.selectbox("Melyik mutatót nézzük?", options=asym_cols_present, format_func=lambda x: asym_labels.get(x, x), key="asym_sel")
            plot_asym = asym_base.dropna(subset=[sel_asym]).copy()
            fig_asym = px.scatter(plot_asym, x="Dátum", y=sel_asym, color="Edzés típusa" if "Edzés típusa" in plot_asym.columns else None, title=f"{asym_labels.get(sel_asym, sel_asym)} idősor", opacity=0.7)
            fig_asym.add_hline(y=CFG["asym_warn_pct"], line_dash="dot", line_color="orange", annotation_text=f"Figyelmeztetés ({CFG['asym_warn_pct']}%)")
            fig_asym.add_hline(y=CFG["asym_red_pct"], line_dash="dot", line_color="red", annotation_text=f"Kockázati küszöb ({CFG['asym_red_pct']}%)")
            st.plotly_chart(fig_asym, use_container_width=True)

# =========================================================
# TAB: VERSENYIDŐ BECSLŐ
# =========================================================
with tab_race:
    st.subheader("🏆 Versenyidő becslő")
    _rc1, _rc2, _rc3 = st.columns(3)
    _race_runs = d[d[run_type_col] == "race"].dropna(subset=["dist_km", "pace_sec_km"]) if run_type_col and "dist_km" in d.columns else pd.DataFrame()
    _known_dist_sel = _rc1.selectbox("Ismert táv", options=list(_RACE_DISTANCES.keys()), index=4, key="race_pred_dist")
    _known_km = _RACE_DISTANCES[_known_dist_sel]
    _auto_time = None
    if not _race_runs.empty:
        _close = _race_runs[(_race_runs["dist_km"] >= _known_km * 0.9) & (_race_runs["dist_km"] <= _known_km * 1.1)].sort_values("Dátum", ascending=False)
        if len(_close) > 0:
            _best = _close.iloc[0]
            _auto_time = int(_best["pace_sec_km"] * _best["dist_km"])
    _default_min = int(_auto_time // 60) if _auto_time else 45
    _default_sec = int(_auto_time % 60) if _auto_time else 0
    _inp_min = _rc2.number_input("Idő – perc", min_value=0, max_value=600, value=_default_min, step=1, key="race_min")
    _inp_sec = _rc3.number_input("Idő – másodperc", min_value=0, max_value=59, value=_default_sec, step=1, key="race_sec")
    _known_time_sec = _inp_min * 60 + _inp_sec
    if _known_time_sec <= 0:
        st.info("Add meg az ismert versenyeredményt.")
    else:
        _known_pace = _known_time_sec / _known_km
        _rc1.metric("Megadott tempó", f"{sec_to_pace_str(_known_pace)} /km")
        _ctl_now = _atl_now = None
        if "TSS_proxy" in d.columns and d["TSS_proxy"].notna().any():
            _acwr_tmp = compute_acwr(d, load_col="TSS_proxy")
            if not _acwr_tmp.empty:
                _pmc_tmp = compute_ctl_atl_tsb(_acwr_tmp)
                if not _pmc_tmp.empty:
                    _lp = _pmc_tmp.iloc[-1]; _ctl_now = float(_lp.get("CTL", np.nan)); _atl_now = float(_lp.get("ATL", np.nan))
        result = race_time_table(_known_dist_sel, _known_time_sec, _ctl_now, _atl_now, float(d[fatigue_col].dropna().iloc[-1]) if (fatigue_col and d[fatigue_col].notna().any()) else None)
        if isinstance(result, tuple):
            _df_race, _forma_txt, _tsb = result
            st.markdown(f"**Forma-korrekció:** {_forma_txt}")
            st.markdown("### 📊 Fő versenytávok")
            _main_dists = ["5 km", "10 km", "Félmaraton", "Maraton"]
            _card_cols = st.columns(4)
            for i, dist_name in enumerate(_main_dists):
                _row = _df_race[_df_race["Táv"] == dist_name]
                if _row.empty: continue
                _r = _row.iloc[0]
                with _card_cols[i]:
                    st.markdown(f'<div style="background:var(--color-background-secondary,#1a1a2e);border-radius:12px;padding:14px 12px;text-align:center;"><div style="font-size:0.8rem;color:#aaa">{dist_name}</div><div style="font-size:1.6rem;font-weight:500">{_r["Forma-korrigált"]}</div><div style="font-size:0.8rem;color:#aaa">{_r["Tempó (min/km)"]} /km</div></div>', unsafe_allow_html=True)
            with st.expander("📋 Összes távolság"):
                st.dataframe(_df_race[["Táv", "Alap becslés", "Forma-korrigált", "Tempó (min/km)"]], use_container_width=True, hide_index=True)
            _vo2 = vo2max_from_race(_known_km, _known_time_sec)
            if pd.notna(_vo2):
                _vo2_cols = st.columns(4)
                _vo2_cat = "Kiváló" if _vo2 >= 60 else "Jó" if _vo2 >= 52 else "Átlag felett" if _vo2 >= 45 else "Átlagos" if _vo2 >= 38 else "Átlag alatt"
                _vo2_cols[0].metric("Becsült VO2max", f"{_vo2:.1f} ml/kg/min")
                _vo2_cols[1].metric("Kategória", _vo2_cat)

# =========================================================
# TAB: HŐMÉRSÉKLET-KORREKCIÓ
# =========================================================
with tab_heat:
    st.subheader("🌡️ Hőmérséklet-korrekció")
    _has_temp = "temp_c" in d.columns and d["temp_c"].notna().any()
    if not _has_temp:
        st.info("Nincs hőmérséklet adat a feltöltött fájlban.")
    else:
        d_heat = compute_heat_adjusted_df(d, hr_col)
        st.markdown("### 📋 Hőmérséklet-korrekciós referencia")
        _ref_rows = []
        for _t in [0, 5, 10, 15, 18, 20, 22, 24, 26, 28, 30, 32, 35]:
            _hc = heat_pace_correction(float(_t))
            _ref_rows.append({"Hőmérséklet (°C)": _t, "Tempó-plusz (sec/km)": f"+{_hc['correction_sec']:.0f}" if _hc["correction_sec"] > 0 else "—", "HR-plusz (bpm)": f"+{_hc['hr_correction']:.1f}" if _hc["hr_correction"] > 0 else "—", "Státusz": {"cold": "❄️ Hideg", "optimal": "✅ Optimális", "warm": "🟡 Meleg", "hot": "🟠 Forró", "very_hot": "🔴 Nagy meleg", "dangerous": "🚨 Veszélyes"}.get(_hc["status"], "—")})
        st.dataframe(pd.DataFrame(_ref_rows), use_container_width=True, hide_index=True)
        _heat_plot = d_heat.dropna(subset=["Dátum", "pace_sec_km"]).copy()
        _heat_plot = _heat_plot[_heat_plot["pace_sec_km"] > 0].copy()
        if len(_heat_plot) >= 3:
            fig_heat = px.scatter(_heat_plot, x="Dátum", y="pace_sec_km", color_discrete_sequence=["#3498db"], opacity=0.5, title="Tényleges vs hőkorrigált tempó")
            fig_heat.update_traces(name="Tényleges tempó", showlegend=True)
            if "pace_heat_adj_sec" in _heat_plot.columns and _heat_plot["pace_heat_adj_sec"].notna().any():
                _adj_plot = _heat_plot.dropna(subset=["pace_heat_adj_sec"])
                fig_heat.add_scatter(x=_adj_plot["Dátum"], y=_adj_plot["pace_heat_adj_sec"], mode="markers", marker=dict(color="#2ecc71", size=6, opacity=0.7), name="Hőkorrigált")
            fig_heat.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# TAB: STRAVA ADATOK
# =========================================================
with tab_strava:
    st.subheader("🟠 Strava adatok – részletes nézet")
    if _data_source == "garmin":
        st.info("Ez a tab Strava csatlakozás esetén aktív.")
    else:
        access_token = st.session_state.get("strava_access_token")
        if access_token:
            last_act = _strava_get("/athlete/activities", access_token, params={"per_page": 5, "page": 1})
            run_acts = [a for a in (last_act or []) if a.get("sport_type", a.get("type", "")) in ("Run", "TrailRun", "VirtualRun", "Race", "Workout")]
            if run_acts:
                options_s = {f"{a.get('start_date_local','')[:10]}  –  {a.get('name','?')}  ({a.get('distance',0)/1000:.1f} km)": a for a in run_acts}
                chosen_s = st.selectbox("Melyik futást nézzük?", options=list(options_s.keys()), key="strava_debug_sel")
                act = options_s[chosen_s]
                _c = st.columns(4)
                _c[0].metric("Távolság", f"{act.get('distance',0)/1000:.2f} km")
                _c[1].metric("Idő", f"{int(act.get('moving_time',0)//60)} perc")
                hr_v = act.get("average_heartrate"); _c[2].metric("Átlag HR", f"{hr_v:.0f} bpm" if hr_v else "—")
                cad_v = act.get("average_cadence"); _c[3].metric("Kadencia", f"{cad_v*2:.0f} spm" if cad_v else "—")
                with st.expander("🔩 Teljes nyers Strava JSON"):
                    st.json({k: v for k, v in act.items() if k != "map"})
            else:
                st.info("Nem találtam futós aktivitást az utolsó 5 aktivitás között.")
        else:
            st.warning("Strava access token nem elérhető.")

# =========================================================
# TAB: STRAVA ELEMZÉS
# =========================================================
with tab_strava_analysis:
    st.subheader("🟠 Strava elemzés – csak Strava adatok alapján")
    if _data_source == "garmin":
        st.info("Ez az elemzés Strava csatlakozás esetén aktív.")
    else:
        _s = view.copy()
        _has_hr   = "hr_num" in _s.columns and _s["hr_num"].notna().sum() >= 3
        _has_pace = "pace_sec_km" in _s.columns and _s["pace_sec_km"].notna().sum() >= 3
        _has_dist = "dist_km" in _s.columns and _s["dist_km"].notna().sum() >= 3
        _has_cad  = "cad_num" in _s.columns and _s["cad_num"].notna().sum() >= 3

        _k = st.columns(5)
        _k[0].metric("Futások", f"{len(_s)}")
        _k[1].metric("Össz távolság", f"{_s['dist_km'].sum():.0f} km" if _has_dist else "—")
        _k[2].metric("Össz idő", f"{_s['dur_sec'].sum()/3600:.1f} h" if "dur_sec" in _s.columns else "—")
        _k[3].metric("Átlag HR", f"{_s['hr_num'].mean():.0f} bpm" if _has_hr else "—")
        _k[4].metric("Össz emelkedés", f"{_s['asc_m'].sum():.0f} m" if "asc_m" in _s.columns and _s["asc_m"].notna().any() else "—")

        if _has_pace and _has_dist:
            st.markdown("### 📈 Tempó fejlődés időben")
            _sp = _s[_s["pace_sec_km"].notna() & (_s["pace_sec_km"] > 0)].copy()
            _sp["pace_inv"] = 1000 / _sp["pace_sec_km"]
            fig_pace = px.scatter(_sp, x="Dátum", y="pace_inv", size="dist_km", size_max=18, color="hr_num" if _has_hr else None, color_continuous_scale="RdYlGn_r", labels={"pace_inv": "Sebesség (km/h)"}, title="Sebesség időben")
            st.plotly_chart(fig_pace, use_container_width=True)

        if _has_dist:
            st.markdown("### 📅 Heti volumen")
            _sw = _s.copy(); _sw["hét"] = _sw["Dátum"].dt.to_period("W").dt.start_time
            _weekly = _sw.groupby("hét").agg(km=("dist_km", "sum"), futások=("dist_km", "count")).reset_index()
            st.plotly_chart(px.bar(_weekly, x="hét", y="km", title="Heti futott kilométerek", labels={"km": "Heti km", "hét": ""}), use_container_width=True)

        if _has_cad:
            st.markdown("### 🦵 Kadencia elemzés")
            _sc = _s[_s["cad_num"].notna() & (_s["cad_num"] > 100)].copy()
            fig_cad = px.scatter(_sc, x="Dátum", y="cad_num", title="Kadencia időben", labels={"cad_num": "Kadencia (spm)"})
            fig_cad.add_hline(y=180, line_dash="dot", line_color="green", annotation_text="180 spm optimum")
            st.plotly_chart(fig_cad, use_container_width=True)


# =========================================================
# TAB: 🤖 AI EDZŐ
# =========================================================
with tab_ai:
    st.subheader("🤖 AI Edző – személyre szabott elemzés")
    st.caption("Claude AI összefoglalja az edzésadataidat és személyre szabott javaslatokat ad.")

    # API kulcs ellenőrzés
    ai_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if not ai_api_key:
        st.error(
            "Hiányzó secret: **ANTHROPIC_API_KEY**\n\n"
            "Add hozzá a Streamlit Secrets-hez:\n"
            "```toml\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```"
        )
    else:
        # --- Kontextus összerakása az adatokból
        last_runs = d.dropna(subset=["Dátum"]).sort_values("Dátum").tail(10)
        recent = d[d["Dátum"] >= (d["Dátum"].max() - pd.Timedelta(weeks=4))]

        def build_ai_context() -> str:
            lines = []
            lines.append(f"Futó adatai: {user_age} éves, {weight_kg} kg, {height_cm} cm, HRmax: {hrmax} bpm")
            lines.append(f"Adatforrás: {_data_source}")
            lines.append(f"Összes futás (szűrt időszak): {len(view)}")

            if "dist_km" in recent.columns and recent["dist_km"].notna().any():
                lines.append(f"Utolsó 4 hét össztáv: {recent['dist_km'].sum():.1f} km")
                lines.append(f"Utolsó 4 hét futások száma: {len(recent)}")

            if "dur_sec" in recent.columns and recent["dur_sec"].notna().any():
                lines.append(f"Utolsó 4 hét összes edzésidő: {recent['dur_sec'].sum()/3600:.1f} óra")

            if "Technika_index" in d.columns and d["Technika_index"].notna().any():
                ti = d["Technika_index"].dropna()
                ti_trend = ti.tail(5).mean() - ti.tail(15).head(10).mean() if len(ti) >= 15 else 0
                lines.append(f"Technika_index – összes átlag: {ti.mean():.1f}, utolsó 5 futás átlag: {ti.tail(5).mean():.1f}, trend: {ti_trend:+.1f}")

            if fatigue_col and d[fatigue_col].notna().any():
                fv = d[fatigue_col].dropna()
                lines.append(f"Fatigue_score – aktuális: {fv.iloc[-1]:.1f}, 4 hetes átlag: {fv.tail(20).mean():.1f}, max az utóbbi időben: {fv.tail(20).max():.1f}")

            if "TSS_proxy" in d.columns and d["TSS_proxy"].notna().any():
                try:
                    _acwr_ctx = compute_acwr(d, load_col="TSS_proxy")
                    if not _acwr_ctx.empty:
                        _pmc_ctx = compute_ctl_atl_tsb(_acwr_ctx)
                        if not _pmc_ctx.empty:
                            lp = _pmc_ctx.iloc[-1]
                            lines.append(f"PMC – CTL (fittség): {lp.get('CTL', '?'):.1f}, ATL (fáradtság): {lp.get('ATL', '?'):.1f}, TSB (forma): {lp.get('TSB', '?'):+.1f} ({lp.get('TSB_status', '?')})")
                            lines.append(f"ACWR: {lp.get('acwr', '?'):.2f} ({lp.get('acwr_status', '?')})" if 'acwr' in lp and pd.notna(lp.get('acwr')) else "ACWR: nincs adat")
                except Exception:
                    pass

            if run_type_col and run_type_col in d.columns:
                type_dist = d.groupby(run_type_col)["dist_km"].agg(["sum", "count"]).round(1)
                for rt, row in type_dist.iterrows():
                    lines.append(f"Edzés típus – {run_type_hu(rt)}: {row['count']:.0f} futás, {row['sum']:.0f} km összesen")

            if "Asymmetry_score" in d.columns and d["Asymmetry_score"].notna().any():
                asym_last = float(d.dropna(subset=["Asymmetry_score"]).sort_values("Dátum").iloc[-1]["Asymmetry_score"])
                lines.append(f"Aszimmetria score (utolsó futás): {asym_last:.1f}%")

            # Utolsó 5 futás részletei
            lines.append("\n--- UTOLSÓ 5 FUTÁS ---")
            for _, row in last_runs.tail(5).iterrows():
                r_type = run_type_hu(row.get("Run_type", "")) or "—"
                r_dist = f"{row.get('dist_km', '?'):.1f} km" if pd.notna(row.get("dist_km")) else "?"
                r_pace = sec_to_pace_str(row.get("pace_sec_km")) + "/km" if pd.notna(row.get("pace_sec_km")) else "?"
                r_hr   = f"{row.get('hr_num', '?'):.0f} bpm" if pd.notna(row.get("hr_num")) else "?"
                r_tech = f"Tech:{row.get('Technika_index', '?'):.0f}" if pd.notna(row.get("Technika_index")) else ""
                r_fat  = f"Fat:{row.get(fatigue_col, '?'):.0f}" if (fatigue_col and pd.notna(row.get(fatigue_col))) else ""
                r_elev = f"↑{row.get('asc_m', 0):.0f}m" if pd.notna(row.get("asc_m")) and row.get("asc_m", 0) > 0 else ""
                lines.append(f"  {row['Dátum'].strftime('%Y-%m-%d')} | {r_type} | {r_dist} | {r_pace} | HR:{r_hr} | {r_tech} {r_fat} {r_elev}".strip())

            return "\n".join(lines)

        ctx = build_ai_context()

        # --- Kérdés típus választó
        st.markdown("### 💬 Mit szeretnél megtudni?")
        ai_mode = st.radio(
            "Válassz kérdéstípust:",
            options=[
                "📊 Átfogó elemzés és értékelés",
                "📅 Holnapi edzésjavaslat",
                "📈 Fejlődési trend és javaslatok",
                "🏆 Verseny-felkészülési tanács",
                "❓ Saját kérdés",
            ],
            horizontal=False,
            key="ai_mode",
        )

        user_question = None
        if ai_mode == "❓ Saját kérdés":
            user_question = st.text_area(
                "Írd be a kérdésedet:",
                placeholder="Pl. Mikor legyek készen egy félmaratonra? Miért romlik a technikám? Mit javítsak először?",
                key="ai_custom_q",
                height=100,
            )

        # --- Generálás gomb
        col_btn, col_tip = st.columns([1, 3])
        with col_btn:
            generate_clicked = st.button("🚀 AI elemzés", type="primary", key="ai_generate", use_container_width=True)
        with col_tip:
            st.caption("Az AI látja az összes edzésadatodat és személyre szabott választ ad.")

        if generate_clicked:
            prompts = {
                "📊 Átfogó elemzés és értékelés": (
                    "Kérlek elemezd az alábbi futó edzésadatait részletesen! "
                    "Adj átfogó értékelést: terhelés trendje, technika állapota, fáradtság szintje, forma (TSB/CTL/ATL). "
                    "Emeld ki a 3 legfontosabb erősséget és a 3 legfontosabb fejlesztendő területet. "
                    "Adj konkrét, cselekvésre ösztönző javaslatokat. Magyarul válaszolj."
                ),
                "📅 Holnapi edzésjavaslat": (
                    "Az alábbi edzésadatok alapján adj konkrét javaslatot a HOLNAPI edzésre! "
                    "Írd le: milyen típusú legyen (easy/tempo/interval/pihenő), mekkora táv, milyen tempó/HR tartomány, mennyi idő. "
                    "Indokold meg röviden az aktuális fáradtság, forma és technika alapján. "
                    "Ha pihenőnap javasolt, magyarázd el miért. Magyarul válaszolj."
                ),
                "📈 Fejlődési trend és javaslatok": (
                    "Elemezd a futó fejlődési trendjét az adatok alapján! "
                    "Van-e javulás a technikában (Technika_index), futás-gazdaságosságban (RES+), formában (TSB)? "
                    "Azonosítsd a hosszú távú mintákat: mikor volt a csúcspont, mi okoz ingadozást. "
                    "Adj 3 konkrét, személyre szabott javaslatot a következő 4-6 hétre. Magyarul válaszolj."
                ),
                "🏆 Verseny-felkészülési tanács": (
                    "Az alábbi edzésadatok alapján adj verseny-előkészülési tanácsot! "
                    "Értékeld a jelenlegi versenyképességet (readiness), a forma alakulását (TSB). "
                    "Javasold az optimális taper stratégiát (mikor és mennyit csökkents), "
                    "az ideális verseny-napi tempót és a legfontosabb tudnivalókat. Magyarul válaszolj."
                ),
            }

            if ai_mode == "❓ Saját kérdés":
                if not user_question or not user_question.strip():
                    st.warning("Írd be a kérdésedet!")
                    st.stop()
                prompt = user_question.strip()
            else:
                prompt = prompts[ai_mode]

            full_prompt = f"{prompt}\n\n{'='*50}\nEDZÉSADATOK:\n{'='*50}\n{ctx}"

            with st.spinner("Claude elemzi az adataidat… (5-15 másodperc)"):
                try:
                    import anthropic as _anthropic
                    _client = _anthropic.Anthropic(api_key=ai_api_key)
                    _resp = _client.messages.create(
                        model="claude-sonnet-4-5-20251001",
                        max_tokens=1500,
                        system=(
                            "Te egy tapasztalt futóedző és sporttudományos szakértő vagy. "
                            "Futásbiomechanikai adatokat, terhelésmenedzsmentet (CTL/ATL/TSB/ACWR) "
                            "és teljesítményindexeket (Technika_index, Fatigue_score, RES+) értesz. "
                            "Mindig konkrét, személyre szabott tanácsokat adsz tudományos alapon. "
                            "Magyarázod a döntések logikáját. Tömör de informatív válaszokat adsz. "
                            "Pozitív, motiváló hangvételű vagy, de őszintén mutatod a kockázatokat is."
                        ),
                        messages=[{"role": "user", "content": full_prompt}],
                    )
                    ai_response = _resp.content[0].text

                    st.markdown("---")
                    st.markdown("### 🤖 AI Edző válasza")
                    st.markdown(ai_response)

                    # Mentés session-be
                    st.session_state["last_ai_response"] = ai_response
                    st.session_state["last_ai_mode"] = ai_mode
                    st.session_state["last_ai_ctx"] = ctx

                except Exception as e:
                    st.error(f"❌ API hiba: {e}\n\nEllenőrizd az ANTHROPIC_API_KEY secret értékét.")

        elif "last_ai_response" in st.session_state:
            st.markdown("---")
            st.markdown(f"### 🤖 Előző elemzés")
            st.caption(f"Kérdéstípus: {st.session_state.get('last_ai_mode', '?')} | Az adatok frissülhettek azóta – kattints a gombra az újrageneráláshoz.")
            st.markdown(st.session_state["last_ai_response"])

        # Adatkontextus megtekintése
        with st.expander("🔍 Milyen adatokat lát az AI?"):
            st.code(ctx, language="text")

# =========================================================
# TAB: ADATOK
# =========================================================
with tab_data:
    st.subheader("📄 Adatok (szűrve)")
    st.dataframe(view, use_container_width=True, hide_index=True, height=520)
