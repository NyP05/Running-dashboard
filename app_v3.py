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
    "ramp_warn": 8.0,
    "ramp_red": 12.0,
    "daily_coach_tech_bad": -5.0,
    "daily_coach_tech_warn": -2.0,
    "daily_coach_fat_bad": 60.0,
    "daily_coach_fat_warn": 45.0,
    "daily_coach_trend_bad": -6.0,
    # ---- ACWR / TSS / CTL / ATL
    "acwr_acute_days": 7,
    "acwr_chronic_days": 28,
    "acwr_safe_lo": 0.8,
    "acwr_safe_hi": 1.3,
    "acwr_warn_hi": 1.5,
    "ctl_decay": 42,
    "atl_decay": 7,
    "tsb_green": 5,
    "tsb_red": -30,
    # ---- Recovery time
    "recovery_lookback_weeks": 24,
    "recovery_min_events": 8,
    # ---- Aszimmetria
    "asym_warn_pct": 3.0,
    "asym_red_pct": 5.0,
    # ---- Optimális terhelés modell
    "optload_lag_weeks": 2,
    "optload_min_weeks": 10,
    "intratio_lag_weeks": 3,
    # ---- Easy Run Target zóna
    "easy_target_hr_pct_lo": 0.68,   # HRmax % – easy zóna alsó határ
    "easy_target_hr_pct_hi": 0.76,   # HRmax % – easy zóna felső határ
    "easy_fatigue_penalty_hr": 3.0,  # bpm levonás minden 10 Fatigue pont után
    "easy_fatigue_penalty_pace": 4.0,# sec/km levonás minden 10 Fatigue pont után
    "easy_fatigue_penalty_pwr": 5.0, # W levonás minden 10 Fatigue pont után
    "easy_target_band_hr": 5,        # ±bpm a célsáv szélessége
    "easy_target_band_pace": 7,      # ±sec/km a célsáv szélessége
    "easy_target_band_pwr": 10,      # ±W a célsáv szélessége
}

# =========================================================
# STRAVA INTEGRÁCIÓ
# =========================================================

# --- Strava OAuth2 konstansok
_STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
_STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
_STRAVA_API_BASE = "https://www.strava.com/api/v3"

# Strava aktivitástípus → Run_type mapping
_STRAVA_TYPE_MAP = {
    "Run": "easy",        # fallback – pace alapján pontosítjuk
    "TrailRun": "easy",
    "VirtualRun": "easy",
    "Race": "race",
    "Workout": "tempo",
}

# Strava mezők → pipeline oszlopok megfeleltetése
# (azokat az oszlopokat képezzük le, amit a pipeline és az elemzők elvárnak)
_STRAVA_FIELD_MAP = {
    "Cím":                    "name",
    "Tevékenység típusa":     "_type",          # belső, konvertáljuk
    "Távolság":               "_dist_m",         # méterben jön, km-re váltjuk
    "Idő":                    "_dur_sec_raw",
    "Átlagos tempó":          "_pace_raw",        # sec/m → min/km
    "Átlagos pulzusszám":     "average_heartrate",
    "Max pulzus":             "max_heartrate",
    "Teljes emelkedés":       "total_elevation_gain",
    "Átl. pedálütem":         "average_cadence",  # Strava 2×-es → osztjuk
    "Átl. teljesítmény":      "average_watts",
    "Max. teljesítmény":      "max_watts",
    "Hőmérséklet":            "average_temp",
    "Dátum":                  "start_date_local",
}


def _strava_secrets() -> tuple[str | None, str | None, str | None]:
    """Visszaadja a (client_id, client_secret, refresh_token) hármast a Secrets-ből."""
    cid   = st.secrets.get("STRAVA_CLIENT_ID", None)
    csec  = st.secrets.get("STRAVA_CLIENT_SECRET", None)
    rtok  = st.secrets.get("STRAVA_REFRESH_TOKEN", None)
    return cid, csec, rtok


def strava_auth_url(client_id: str, redirect_uri: str) -> str:
    """Generálja az OAuth2 authorization URL-t (csak a helper scripthez kell)."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": "read,activity:read_all",
    }
    return f"{_STRAVA_AUTH_URL}?{urllib.parse.urlencode(params)}"


def strava_exchange_code(client_id: str, client_secret: str, code: str) -> dict | None:
    """Authorization code → access + refresh token csere (helper scripthez)."""
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
    """Lejárt access token megújítása refresh token-nel."""
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
    """Egyszerű Strava API GET hívás. Rate-limit kezeléssel."""
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
    """
    Letölti az utolsó `days_back` nap futó aktivitásait a Strava API-ból.
    Cache: 30 perc (ttl=1800).
    """
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
        # Csak futós aktivitások
        runs = [a for a in batch if a.get("sport_type", a.get("type", "")) in
                ("Run", "TrailRun", "VirtualRun", "Race", "Workout")]
        all_acts.extend(runs)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.3)   # udvarias rate-limit

    return all_acts


def strava_activities_to_df(activities: list[dict]) -> pd.DataFrame:
    """
    Strava aktivitás lista → pipeline-kompatibilis DataFrame.

    Konverziók:
    - távolság: m → km
    - tempó: average_speed (m/s) → min/km string + sec/km numerikus
    - kadencia: Strava 2× (jobb láb) → 2×-re hagyja, a pipeline is így kapja
    - emelkedés: total_elevation_gain → asc_m (des_m: nincs a Strava-ban → NaN)
    - Dátum: ISO8601 string → pd.Timestamp
    """
    if not activities:
        return pd.DataFrame()

    rows = []
    for a in activities:
        dist_m = float(a.get("distance", 0) or 0)
        dist_km = dist_m / 1000.0

        dur_sec = float(a.get("moving_time", 0) or 0)

        # Tempó: m/s → sec/km → min:sec string
        avg_speed = float(a.get("average_speed", 0) or 0)
        if avg_speed > 0:
            pace_sec_km = 1000.0 / avg_speed
            pace_str = sec_to_pace_str(pace_sec_km)
        else:
            pace_sec_km = np.nan
            pace_str = np.nan

        # Dátum – tz-naive-ra normalizálva (Garmin CSV-vel való kompatibilitáshoz)
        date_raw = a.get("start_date_local", a.get("start_date", ""))
        try:
            dt = pd.to_datetime(date_raw, utc=False, errors="coerce")
            if dt is not pd.NaT and dt.tzinfo is not None:
                dt = dt.tz_convert("UTC").tz_localize(None)
        except Exception:
            dt = pd.NaT

        # Kadencia (Strava: avg_cadence = lépések/perc/2 = jobb lábak → *2 = összes)
        cad_raw = a.get("average_cadence")
        cad_num = float(cad_raw) * 2 if cad_raw is not None else np.nan

        # HR
        hr_num = float(a.get("average_heartrate") or np.nan) if a.get("average_heartrate") else np.nan

        # Power (Stryd / Garmin Running Power)
        power_avg = float(a.get("average_watts") or np.nan) if a.get("average_watts") else np.nan
        power_max = float(a.get("max_watts") or np.nan) if a.get("max_watts") else np.nan

        # Emelkedés
        asc_m = float(a.get("total_elevation_gain") or 0)

        # Aktivitás típus → Run_type (pace alapján pontosítjuk később a pipeline-ban)
        sport_type = a.get("sport_type", a.get("type", "Run"))
        run_type_raw = _STRAVA_TYPE_MAP.get(sport_type, "easy")

        rows.append({
            # Pipeline által elvárt oszlopok
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
            "des_m": np.nan,          # Strava nem adja meg
            "Teljes emelkedés": asc_m,
            "Teljes süllyedés": np.nan,
            "temp_c": float(a.get("average_temp") or np.nan) if a.get("average_temp") else np.nan,
            "Run_type": run_type_raw,
            # Strava-specifikus extra mezők (elemzőkben hasznosak)
            "strava_id": a.get("id"),
            "strava_type": sport_type,
            "suffer_score": a.get("suffer_score"),
            "kudos_count": a.get("kudos_count"),
            # Nincs Strava-ban (Garmin Running Dynamics):
            "vr_num": np.nan,
            "gct_num": np.nan,
            "vo_num": np.nan,
            "stride_num": np.nan,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Dátum").reset_index(drop=True)

    # up/down m per km
    df["up_m_per_km"] = np.where(
        df["dist_km"].notna() & (df["dist_km"] > 0),
        df["asc_m"] / df["dist_km"], np.nan
    )
    df["down_m_per_km"] = np.nan
    df["net_elev_m"] = df["asc_m"]

    # slope_bucket (csak emelkedés alapján, süllyedés hiányában)
    df["slope_bucket"] = df["up_m_per_km"].apply(
        lambda u: "flat" if pd.isna(u) or u < 5
        else "rolling" if u < 15
        else "uphill_dominant"
    )

    # temp_bin
    df["temp_bin"] = df["temp_c"].apply(temp_bin)

    # Power_fatigue_hint
    df["Power_fatigue_hint"] = np.where(
        df["power_avg_w"].notna() & df["power_max_w"].notna() & (df["power_avg_w"] > 0),
        df["power_max_w"] / df["power_avg_w"], np.nan
    )

    return df


def get_valid_strava_token(client_id: str, client_secret: str, refresh_tok: str) -> str | None:
    """
    Visszaadja az érvényes access_token-t.
    - Ha a session_state-ben van és még érvényes → visszaadja
    - Ha lejárt vagy nincs → refresh_token-nel megújítja
    A refresh_token a Secrets-ből jön (állandó), az access_token session-szintű.
    """
    ss = st.session_state
    expires_at = ss.get("strava_token_expires_at", 0)
    now_ts = int(time.time())

    if "strava_access_token" in ss and now_ts < expires_at - 300:
        return ss["strava_access_token"]

    # Megújítás a Secrets-beli refresh_token-nel
    new_tokens = strava_refresh_token(client_id, client_secret, refresh_tok)
    if not new_tokens or "access_token" not in new_tokens:
        return None

    ss["strava_access_token"]      = new_tokens["access_token"]
    ss["strava_token_expires_at"]  = new_tokens["expires_at"]
    # Ha a Strava új refresh_token-t ad (ritka), azt is mentjük session-be
    if "refresh_token" in new_tokens:
        ss["strava_session_refresh"] = new_tokens["refresh_token"]

    return ss["strava_access_token"]


def render_strava_connect_sidebar():
    """
    Strava kapcsolat UI a sidebarban – refresh_token alapú (nincs OAuth redirect).

    Állapotok:
      - Nincs secret → útmutató a get_strava_token.py scripthez
      - Van secret, de token hiba → hibaüzenet
      - Minden OK → kapcsolt állapot + szinkron gomb
    """
    client_id, client_secret, refresh_tok = _strava_secrets()

    st.sidebar.divider()
    st.sidebar.header("🟠 Strava kapcsolat")

    # ---- Nincs secret
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

    # ---- Token megszerzése / megújítása
    # Session-szintű refresh_token (ha a Strava újat adott) vagy a Secrets-beli
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

    # ---- Kapcsolt állapot
    # Athlete neve: ha még nem töltöttük le, lekérjük egyszer
    if "strava_athlete_name" not in st.session_state:
        athlete_data = _strava_get("/athlete", access_token)
        if athlete_data:
            st.session_state["strava_athlete_name"] = (
                f"{athlete_data.get('firstname', '')} "
                f"{athlete_data.get('lastname', '')}".strip()
            )

    athlete_name = st.session_state.get("strava_athlete_name", "Strava sportoló")
    st.sidebar.success(f"✅ Kapcsolódva: **{athlete_name}**")

    # Szinkron időszak
    days_back = st.sidebar.selectbox(
        "Szinkron időszak",
        options=[90, 180, 365, 730],
        index=2,
        format_func=lambda d: {90: "3 hónap", 180: "6 hónap",
                                365: "1 év",   730: "2 év"}[d],
        key="strava_days_back",
    )

    # Kézi frissítés gomb
    if st.sidebar.button("🔄 Strava adatok frissítése", key="strava_refresh_btn"):
        fetch_strava_activities.clear()
        st.session_state.pop("strava_access_token", None)  # force token újra
        st.rerun()

    # Letöltés (30 perces cache)
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
    # Ha már numerikus dtype (float/int), nincs szükség konverzióra
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    # Object/string: biztonságos konverzió .str nélkül
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
# Irodalmi alapok (Gottschall & Kram 2005, Snyder & Farley 2011):
#   - Emelkedőn: VO csökken (~2%/5m·km⁻¹), GCT csökken, lépéshossz nő
#   - Lejtőn: VO nő, GCT nő, lépéshossz csökken
#   - VR: emelkedőn kismértékben csökken
#   - Kadencia: relatíve stabil (±1%/10m·km⁻¹)
# A korrekció célja: a slope hatását "kivonni" az indexekből,
# így a Technika/Fatigue/RES értékek terep-függetlenek lesznek.

# Korrekciós együtthatók (minden egység: hatás per m/km emelkedés)
# Pozitív = az érték nő emelkedőn (pl. VO nő → romlás látszata)
# Negatív = az érték csökken emelkedőn (pl. GCT csökken → javulás látszata)
_SLOPE_CORR = {
    # col        up_coeff    down_coeff  (per m/km)
    "vo_num":   (+0.040,     -0.030),   # VO: emelkedőn CSÖKKEN, lejtőn NŐ
    "gct_num":  (-0.300,     +0.200),   # GCT: emelkedőn csökken (ms/m·km⁻¹)
    "vr_num":   (-0.010,     +0.008),   # VR: emelkedőn kicsit csökken (%)
    "stride_num": (+0.002,   -0.002),   # Lépéshossz: emelkedőn nő (m)
    "cad_num":  (-0.008,     +0.006),   # Kadencia: minimális változás
    "hr_num":   (+0.150,     -0.080),   # HR: emelkedőn nő (bpm/m·km⁻¹)
    "pace_sec_km": (-2.5,    +1.8),     # Tempó: emelkedőn gyorsabb sec/km érték
                                        # (kisebb szám = gyorsabb, ezért negatív)
}


def slope_correct_series(
    series: pd.Series,
    up_m_per_km: pd.Series,
    down_m_per_km: pd.Series,
    col_name: str,
) -> pd.Series:
    """
    Egy adatoszlopból kivonja a terep hatását.
    Visszaadja a "sík-ekvivalens" értéket.

    Ha col_name nincs a _SLOPE_CORR-ban, változtatás nélkül adja vissza.
    """
    if col_name not in _SLOPE_CORR:
        return series
    up_c, down_c = _SLOPE_CORR[col_name]
    up = up_m_per_km.fillna(0).clip(lower=0)
    down = down_m_per_km.fillna(0).clip(lower=0)
    correction = up_c * up + down_c * down
    return series - correction


def apply_slope_correction(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    DataFrame-re alkalmazza a slope-korrekciót a megadott (vagy alapértelmezett)
    oszlopokra. Új oszlopokat hoz létre `_sc` (slope-corrected) suffix-szel.

    Pl.: vo_num → vo_num_sc, gct_num → gct_num_sc
    """
    out = df.copy()
    if "up_m_per_km" not in out.columns or "down_m_per_km" not in out.columns:
        # Ha nincs terep adat, visszaadja az eredetit (suffix nélkül = alias)
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
    """
    Megadja, hogy egy adott terep-profilnál mekkora korrekció várható
    az egyes mutatókra. Debug / magyarázó célra.
    """
    up = max(0.0, float(up_m_per_km) if pd.notna(up_m_per_km) else 0.0)
    down = max(0.0, float(down_m_per_km) if pd.notna(down_m_per_km) else 0.0)
    result = {}
    for col, (uc, dc) in _SLOPE_CORR.items():
        corr = uc * up + dc * down
        result[col] = round(corr, 3)
    result["_up_m_per_km"] = up
    result["_down_m_per_km"] = down
    return result


def _safe_dropna(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """
    dropna mint a pandas-é, de csak a ténylegesen létező oszlopokra alkalmazza.
    Ha egy oszlop nem létezik a df-ben, azt a szűrési feltételt kihagyja.
    Strava módban (pl. Technika_index hiányzik) így nem dob KeyError-t,
    és az eredmény egy üres df lesz ha az egyetlen subset-col hiányzik.
    """
    existing = [c for c in subset if c in df.columns]
    missing  = [c for c in subset if c not in df.columns]
    if missing:
        # Ha legalább egy kötelező oszlop hiányzik teljesen,
        # adjunk vissza üres df-et (nem tudunk érvényes sorokat szűrni)
        return df.iloc[0:0].copy()
    return df.dropna(subset=existing)


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
    easy = _safe_dropna(easy, ["Dátum", "Technika_index"]).sort_values("Dátum")
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

    # Strava módban Technika_index nem létezik → nincs coach üzenet
    if "Technika_index" not in base_all.columns or base_all["Technika_index"].notna().sum() < 5:
        return "ℹ️", "Technika_index nem elérhető (Strava módban GCT/VO/VR hiányzik). Tölts fel Garmin exportot is a teljes elemzéshez."

    b = _safe_dropna(base_all, ["Dátum", "Technika_index"]).sort_values("Dátum")
    if len(b) < 5:
        return "ℹ️", "Nincs elég technika adat (legalább ~5 futás kell)."
    easy = (
        b[b[run_type_col] == "easy"].copy()
        if run_type_col and run_type_col in b.columns
        else b.copy()
    )
    easy = _safe_dropna(easy, ["Dátum", "Technika_index"]).sort_values("Dátum")
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
# EASY RUN TARGET MODUL
# =========================================================

def compute_easy_target(
    df: pd.DataFrame,
    hrmax: int,
    fatigue_col: str | None,
    run_type_col: str | None,
    hr_col: str | None,
    pace_col: str | None,
    pwr_col: str | None,
    baseline_weeks: int = 12,
    baseline_min_runs: int = 20,
) -> dict | None:
    """
    Napi Easy Run Target zóna számítása.

    Logika (3 réteg):

    1) ALAP ZÓNA (Fatmax + HR%):
       - Fatmax HR-ből kiindulva képezzük az easy zónát
         (Fatmax alatt tartjuk magunkat, de nem túl alacsony)
       - Ha nincs Fatmax, HRmax % alapú fallback (68–76%)

    2) SZEMÉLYRE SZABÁS (saját easy futások mediánja):
       - Az utolsó baseline_weeks hét easy futásainak medián HR/pace/power értéke
         adja a "személyes easy baseline"-t
       - Ez fontosabb mint a generikus % (mert ténylegesen téged jellemez)

    3) FÁRADTSÁG-KORREKCIÓ (Fatigue_score alapján):
       - Minden 10 Fatigue pont felett a 40-es "nyugalmi" szint fölött:
         HR célt LEJJEBB toljuk (a szív nehezebben dolgozik, óvjuk)
         Pace célt LASSABB irányba toljuk
         Power célt LEJJEBB toljuk
       - Ez a napi "frissesség" hatás

    Visszatér: dict a célsávokkal és az összes közbülső értékkel.
    """
    if df is None or len(df) == 0:
        return None

    base = df.dropna(subset=["Dátum"]).sort_values("Dátum").copy()

    # ---- 1. Easy baseline futások kiválasztása
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

    if len(easy_bl) < 5:
        return None

    # ---- 2. Személyes easy medián értékek
    hr_med = float(np.nanmedian(easy_bl[hr_col])) if hr_col and hr_col in easy_bl.columns else np.nan
    pace_med_sec = float(np.nanmedian(easy_bl["pace_sec_km"])) if "pace_sec_km" in easy_bl.columns else np.nan
    pwr_med = float(np.nanmedian(easy_bl[pwr_col])) if pwr_col and pwr_col in easy_bl.columns else np.nan

    # ---- 3. Fatmax alapú HR zóna
    fatmax_hr = np.nan
    fatmax_pace_sec = np.nan
    fatmax_pwr = np.nan

    if hr_col and "pace_sec_km" in base.columns:
        # Fatmax-hoz az easy baseline-t használjuk
        tmp_fm = easy_bl.copy()
        if "Átlagos tempó" not in tmp_fm.columns and "pace_sec_km" in tmp_fm.columns:
            # pace_sec_km → visszakonvertálás stringgé (estimate_fatmax ezt várja)
            tmp_fm["_pace_str"] = tmp_fm["pace_sec_km"].apply(
                lambda s: sec_to_pace_str(s) if pd.notna(s) else np.nan
            )
            _pace_col_fm = "_pace_str"
        else:
            _pace_col_fm = pace_col if pace_col else "Átlagos tempó"

        fm = estimate_fatmax_from_runs(
            tmp_fm, hrmax=hrmax,
            hr_col=hr_col,
            pace_col=_pace_col_fm,
            pwr_col=pwr_col,
            min_points=10,
        )
        if fm:
            fatmax_hr = fm["hr_fatmax"]
            fatmax_pwr = fm["pwr_fatmax"] if pd.notna(fm["pwr_fatmax"]) else np.nan
            # Fatmax pace sec visszakonvertálás
            fatmax_pace_sec = pace_str_to_sec(fm["pace_fatmax"])

    # ---- 4. Célpont meghatározása (Fatmax vagy személyes medián)
    # HR: ha van Fatmax, az a felső plafon; ha nincs, HRmax%-alapú
    if pd.notna(fatmax_hr):
        # Easy = Fatmax alatt, ~5–8%-kal lassabb mint Fatmax
        hr_center = fatmax_hr * 0.94
    elif pd.notna(hr_med):
        hr_center = hr_med
    else:
        hr_center = hrmax * (CFG["easy_target_hr_pct_lo"] + CFG["easy_target_hr_pct_hi"]) / 2

    # Pace: ha van Fatmax tempó, abból számolunk (kicsit lassabb)
    if pd.notna(fatmax_pace_sec):
        pace_center_sec = fatmax_pace_sec * 1.06  # ~6% lassabb mint Fatmax
    elif pd.notna(pace_med_sec):
        pace_center_sec = pace_med_sec
    else:
        pace_center_sec = np.nan

    # Power: ha van Fatmax, kicsit kisebb power
    if pd.notna(fatmax_pwr):
        pwr_center = fatmax_pwr * 0.92
    elif pd.notna(pwr_med):
        pwr_center = pwr_med
    else:
        pwr_center = np.nan

    # ---- 5. Fáradtság-korrekció
    current_fatigue = np.nan
    fatigue_penalty_factor = 0.0  # 0 = nincs korrekció

    if fatigue_col and fatigue_col in base.columns and base[fatigue_col].notna().any():
        last_run = base.dropna(subset=[fatigue_col]).iloc[-1]
        current_fatigue = float(last_run[fatigue_col])

        # Alapszint: 40 alatti fatigue = nincs módosítás
        # Minden 10 pont felett → egyre konzervatívabb zóna
        fatigue_above_base = max(0.0, current_fatigue - 40.0)
        fatigue_penalty_factor = fatigue_above_base / 10.0

    hr_penalty = CFG["easy_fatigue_penalty_hr"] * fatigue_penalty_factor
    pace_penalty = CFG["easy_fatigue_penalty_pace"] * fatigue_penalty_factor
    pwr_penalty = CFG["easy_fatigue_penalty_pwr"] * fatigue_penalty_factor

    # Korrigált középérték
    hr_adj = hr_center - hr_penalty
    pace_adj_sec = pace_center_sec + pace_penalty if pd.notna(pace_center_sec) else np.nan
    pwr_adj = pwr_center - pwr_penalty if pd.notna(pwr_center) else np.nan

    # ---- 6. Célsávok (±band)
    band_hr = CFG["easy_target_band_hr"]
    band_pace = CFG["easy_target_band_pace"]
    band_pwr = CFG["easy_target_band_pwr"]

    hr_lo = round(hr_adj - band_hr)
    hr_hi = round(hr_adj + band_hr)

    pace_lo_sec = pace_adj_sec - band_pace if pd.notna(pace_adj_sec) else np.nan
    pace_hi_sec = pace_adj_sec + band_pace if pd.notna(pace_adj_sec) else np.nan

    pwr_lo = round(pwr_adj - band_pwr) if pd.notna(pwr_adj) else np.nan
    pwr_hi = round(pwr_adj + band_pwr) if pd.notna(pwr_adj) else np.nan

    # ---- 7. HRmax% ellenőrzés (ne menjen ki az aerob tartományból)
    hr_lo = max(hr_lo, round(hrmax * CFG["easy_target_hr_pct_lo"]))
    hr_hi = min(hr_hi, round(hrmax * CFG["easy_target_hr_pct_hi"]))

    # ---- 8. Státusz és napszöveg
    if pd.notna(current_fatigue):
        if current_fatigue >= 70:
            readiness_status = "🔴"
            readiness_text = "Nagy fáradtság – nagyon konzervatív easy, vagy pihenőnap"
        elif current_fatigue >= 50:
            readiness_status = "🟠"
            readiness_text = "Közepes fáradtság – legyen valóban könnyű, ne csábulj gyorsabb tempóra"
        elif current_fatigue >= 30:
            readiness_status = "🟡"
            readiness_text = "Normál állapot – klasszikus easy, a zónát tartsd be"
        else:
            readiness_status = "🟢"
            readiness_text = "Friss állapot – a zóna betartásával akár kicsit több km is mehet"
    else:
        readiness_status = "⚪"
        readiness_text = "Fatigue adat nélkül ez a személyes easy medián alapján számolt zóna"

    return {
        # Célsávok
        "hr_lo": hr_lo,
        "hr_hi": hr_hi,
        "pace_lo": sec_to_pace_str(pace_lo_sec),
        "pace_hi": sec_to_pace_str(pace_hi_sec),
        "pwr_lo": pwr_lo,
        "pwr_hi": pwr_hi,
        # Középértékek (debug / részletek)
        "hr_center": round(hr_adj, 1),
        "pace_center": sec_to_pace_str(pace_adj_sec),
        "pwr_center": round(pwr_adj, 1) if pd.notna(pwr_adj) else np.nan,
        # Forrásadatok
        "fatmax_hr": fatmax_hr,
        "fatmax_pace": sec_to_pace_str(fatmax_pace_sec),
        "fatmax_pwr": fatmax_pwr,
        "hr_med_easy": hr_med,
        "pace_med_easy": sec_to_pace_str(pace_med_sec),
        "pwr_med_easy": pwr_med,
        # Korrekció
        "current_fatigue": current_fatigue,
        "fatigue_penalty_factor": round(fatigue_penalty_factor, 2),
        "hr_penalty": round(hr_penalty, 1),
        "pace_penalty_sec": round(pace_penalty, 1),
        "pwr_penalty": round(pwr_penalty, 1),
        # Státusz
        "readiness_status": readiness_status,
        "readiness_text": readiness_text,
        # Historikus trend (utolsó 30 easy futás HR vs zóna)
        "easy_history": easy_bl.tail(30)[
            [c for c in ["Dátum", hr_col, "pace_sec_km", pwr_col, fatigue_col]
             if c and c in easy_bl.columns]
        ].copy() if len(easy_bl) >= 5 else pd.DataFrame(),
    }


# =========================================================
# ÚJ MODUL 1: TSS PROXY
# =========================================================

def compute_tss_proxy(df: pd.DataFrame, hrmax: int) -> pd.Series:
    """
    Training Stress Score proxy futásonként (HR + duration alapján).
    TSS = (dur_sec × hr_avg × IF) / (hrmax × 3600) × 100
    IF = hr_avg / hr_threshold  (hr_threshold = 0.85 × hrmax)
    Fallback ha nincs HR: pace-alapú skálázás.
    """
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


def compute_acwr(df: pd.DataFrame,
                 load_col: str = "TSS_proxy",
                 acute_days: int = 7,
                 chronic_days: int = 28) -> pd.DataFrame:
    """
    ACWR (Acute:Chronic Workload Ratio) napi szinten.
    """
    if load_col not in df.columns or df[load_col].isna().all():
        return pd.DataFrame()

    daily = (
        df.dropna(subset=["Dátum"])
        .groupby(df["Dátum"].dt.date)[load_col]
        .sum()
        .reset_index()
        .rename(columns={"Dátum": "date", load_col: "load"})
        .sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
    daily.columns = ["date", "load"]

    daily["acute"] = daily["load"].rolling(acute_days, min_periods=1).mean()
    daily["chronic"] = daily["load"].rolling(chronic_days, min_periods=7).mean()
    daily["acwr"] = np.where(
        daily["chronic"].notna() & (daily["chronic"] > 0),
        daily["acute"] / daily["chronic"],
        np.nan,
    )

    def _acwr_status(v):
        if pd.isna(v): return "ismeretlen"
        if v < CFG["acwr_safe_lo"]: return "alulterhelt"
        if v <= CFG["acwr_safe_hi"]: return "optimális"
        if v <= CFG["acwr_warn_hi"]: return "figyelmeztető"
        return "veszélyes"

    daily["acwr_status"] = daily["acwr"].apply(_acwr_status)
    return daily


def compute_ctl_atl_tsb(daily_load: pd.DataFrame) -> pd.DataFrame:
    """
    CTL (fittség, ~42 nap EMA), ATL (fáradtság, ~7 nap EMA), TSB (forma = CTL - ATL).
    """
    if daily_load.empty or "load" not in daily_load.columns:
        return pd.DataFrame()

    ctl_d = CFG["ctl_decay"]
    atl_d = CFG["atl_decay"]
    alpha_ctl = 1.0 - np.exp(-1.0 / ctl_d)
    alpha_atl = 1.0 - np.exp(-1.0 / atl_d)

    loads = daily_load["load"].fillna(0).to_numpy(dtype=float)
    ctl = np.zeros(len(loads))
    atl = np.zeros(len(loads))
    ctl[0] = loads[0]
    atl[0] = loads[0]
    for i in range(1, len(loads)):
        ctl[i] = ctl[i - 1] + alpha_ctl * (loads[i] - ctl[i - 1])
        atl[i] = atl[i - 1] + alpha_atl * (loads[i] - atl[i - 1])

    out = daily_load.copy()
    out["CTL"] = ctl
    out["ATL"] = atl
    out["TSB"] = ctl - atl

    def _tsb_status(v):
        if pd.isna(v): return "ismeretlen"
        if v >= CFG["tsb_green"]: return "forma"
        if v >= -10: return "friss"
        if v >= -20: return "fáradt"
        if v >= CFG["tsb_red"]: return "túlterhelt"
        return "kritikus"

    out["TSB_status"] = out["TSB"].apply(_tsb_status)
    return out


def injury_risk_score(
    acwr_val: float | None,
    tsb_val: float | None,
    fatigue_val: float | None,
    asym_val: float | None,
    ramp_val: float | None,
) -> tuple[float, str, list[str]]:
    """
    Összetett sérüléskockázati score (0–100).
    Súlyok: ACWR 35%, TSB 25%, Fatigue 20%, Ramp 15%, Aszimmetria 5%.
    """
    components = []
    explanations = []

    if acwr_val is not None and not np.isnan(float(acwr_val)):
        v = float(acwr_val)
        if v < CFG["acwr_safe_lo"]:
            r, txt = 20, f"ACWR {v:.2f}: alulterhelt (detraining kockázat)"
        elif v <= CFG["acwr_safe_hi"]:
            r, txt = 5, f"ACWR {v:.2f}: optimális zónában ✓"
        elif v <= CFG["acwr_warn_hi"]:
            r, txt = 50, f"ACWR {v:.2f}: figyelmeztető – csökkentsd a terhelést"
        else:
            r, txt = 90, f"ACWR {v:.2f}: veszélyes – azonnali tehercsökkentés"
        components.append((r, 0.35)); explanations.append(txt)

    if tsb_val is not None and not np.isnan(float(tsb_val)):
        v = float(tsb_val)
        if v >= CFG["tsb_green"]:
            r, txt = 10, f"TSB {v:+.1f}: jó forma ✓"
        elif v >= -10:
            r, txt = 25, f"TSB {v:+.1f}: friss"
        elif v >= -20:
            r, txt = 55, f"TSB {v:+.1f}: fáradt"
        elif v >= CFG["tsb_red"]:
            r, txt = 80, f"TSB {v:+.1f}: túlterhelt – pihenő szükséges"
        else:
            r, txt = 95, f"TSB {v:+.1f}: kritikus – azonnali leállás"
        components.append((r, 0.25)); explanations.append(txt)

    if fatigue_val is not None and not np.isnan(float(fatigue_val)):
        v = float(fatigue_val)
        r = float(np.clip(v, 0, 100))
        txt = (f"Fatigue {v:.0f}: magas" if v > 70
               else f"Fatigue {v:.0f}: közepes" if v > 50
               else f"Fatigue {v:.0f}: alacsony ✓")
        components.append((r, 0.20)); explanations.append(txt)

    if ramp_val is not None and not np.isnan(float(ramp_val)):
        v = float(ramp_val)
        r = 10 if v <= 8 else 45 if v <= 12 else 75 if v <= 20 else 95
        if v > 8:
            explanations.append(f"Ramp rate {v:+.1f}%: gyors terhelésnövelés")
        components.append((r, 0.15))

    if asym_val is not None and not np.isnan(float(asym_val)):
        v = float(asym_val)
        if v < CFG["asym_warn_pct"]:
            r = 5
        elif v < CFG["asym_red_pct"]:
            r, _ = 40, explanations.append(f"Aszimmetria {v:.1f}%: figyelemre méltó")
        else:
            r, _ = 75, explanations.append(f"Aszimmetria {v:.1f}%: magas – sérülésprediktív")
        # r lehet None ha append-ből jött, fix:
        r = r if isinstance(r, (int, float)) else 40
        components.append((r, 0.05))

    if not components:
        return np.nan, "ismeretlen", ["Nincs elég adat."]

    total_w = sum(w for _, w in components)
    score = float(np.clip(sum(r * w for r, w in components) / total_w, 0, 100))

    status = (
        "🟢 Alacsony" if score < 25
        else "🟡 Közepes" if score < 50
        else "🟠 Magas" if score < 70
        else "🔴 Kritikus"
    )
    return score, status, [e for e in explanations if e is not None]


# =========================================================
# ÚJ MODUL 2: RECOVERY TIME MODELL
# =========================================================

def compute_recovery_model(
    df: pd.DataFrame,
    fatigue_col: str,
    run_type_col: str | None,
    lookback_weeks: int = 24,
    min_events: int = 8,
) -> dict | None:
    """
    Meghatározza, hogy egy magas Fatigue esemény után hány napra állt vissza
    a Technika_index. Lineáris regressziót illeszt: fatigue → recovery_days.
    """
    if "Technika_index" not in df.columns or fatigue_col not in df.columns:
        return None
    base = _safe_dropna(df, ["Dátum", "Technika_index", fatigue_col]).sort_values("Dátum").copy()
    if len(base) < min_events * 3:
        return None

    cutoff = base["Dátum"].max() - pd.Timedelta(weeks=lookback_weeks)
    base = base[base["Dátum"] >= cutoff].copy()

    easy = (
        base[base[run_type_col] == "easy"].copy()
        if run_type_col and run_type_col in base.columns
        else base.copy()
    )
    if len(easy) < min_events:
        return None

    events = base[base[fatigue_col] > 65].copy()
    if len(events) < min_events:
        thr = np.nanpercentile(base[fatigue_col], 75)
        events = base[base[fatigue_col] >= thr].copy()
    if len(events) < 4:
        return None

    rec_days_list, fat_list, pre_tech_list = [], [], []

    for _, ev in events.iterrows():
        ev_date = ev["Dátum"]
        fat_val = float(ev[fatigue_col])

        pre_w = easy[
            (easy["Dátum"] >= ev_date - pd.Timedelta(days=21))
            & (easy["Dátum"] < ev_date - pd.Timedelta(days=1))
        ]
        if len(pre_w) < 2:
            continue
        pre_tech = float(np.nanmedian(pre_w["Technika_index"]))

        post = easy[
            (easy["Dátum"] > ev_date) & (easy["Dátum"] <= ev_date + pd.Timedelta(days=30))
        ].sort_values("Dátum")
        if len(post) < 2:
            continue

        recovered_day = None
        for i in range(len(post) - 1):
            t1 = float(post.iloc[i]["Technika_index"])
            t2 = float(post.iloc[i + 1]["Technika_index"])
            if abs(t1 - pre_tech) <= 7 and abs(t2 - pre_tech) <= 7:
                recovered_day = (post.iloc[i]["Dátum"] - ev_date).days
                break

        if recovered_day is not None and 0 < recovered_day <= 30:
            rec_days_list.append(recovered_day)
            fat_list.append(fat_val)
            pre_tech_list.append(pre_tech)

    if len(rec_days_list) < 4:
        return None

    rec_arr = np.array(rec_days_list, dtype=float)
    fat_arr = np.array(fat_list, dtype=float)

    coeffs = np.polyfit(fat_arr, rec_arr, 1) if np.std(fat_arr) > 1e-6 else [0, float(np.median(rec_arr))]

    current_fat = None
    if df[fatigue_col].notna().any():
        current_fat = float(df.dropna(subset=[fatigue_col]).sort_values("Dátum").iloc[-1][fatigue_col])

    predicted_days = None
    if current_fat is not None:
        predicted_days = max(0, int(round(float(np.polyval(coeffs, current_fat)))))

    return {
        "rec_arr": rec_arr,
        "fat_arr": fat_arr,
        "coeffs": coeffs,
        "median_recovery": float(np.median(rec_arr)),
        "n_events": len(rec_arr),
        "current_fat": current_fat,
        "predicted_days": predicted_days,
        "df": pd.DataFrame({"fatigue": fat_arr, "recovery_days": rec_arr}),
    }


# =========================================================
# ÚJ MODUL 3: ASZIMMETRIA
# =========================================================

def compute_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bal/jobb aszimmetria számítás Garmin Running Dynamics adatokból.
    Aszimmetria index = |bal - jobb| / ((bal + jobb) / 2) × 100
    """
    result = df.copy()

    def _asym_pct(left_s: pd.Series, right_s: pd.Series) -> pd.Series:
        mean_lr = (left_s + right_s) / 2
        return np.where(
            left_s.notna() & right_s.notna() & (mean_lr > 0),
            np.abs(left_s - right_s) / mean_lr * 100,
            np.nan,
        )

    # GCT
    lc = next((c for c in df.columns if "bal" in c.lower() and "gct" in c.lower()), None)
    rc = next((c for c in df.columns if "jobb" in c.lower() and "gct" in c.lower()), None)
    result["asym_gct_pct"] = (
        pd.Series(_asym_pct(to_float_series(df[lc]), to_float_series(df[rc])), index=df.index)
        if lc and rc else np.nan
    )

    # Lépéshossz
    lc = next((c for c in df.columns if "bal" in c.lower() and ("lépés" in c.lower() or "stride" in c.lower())), None)
    rc = next((c for c in df.columns if "jobb" in c.lower() and ("lépés" in c.lower() or "stride" in c.lower())), None)
    result["asym_stride_pct"] = (
        pd.Series(_asym_pct(to_float_series(df[lc]), to_float_series(df[rc])), index=df.index)
        if lc and rc else np.nan
    )

    # Power balance
    bal_col = next((c for c in df.columns if "egyensúly" in c.lower() or "balance" in c.lower()), None)
    result["asym_power_pct"] = (
        np.abs(to_float_series(df[bal_col]) - 50.0) * 2 if bal_col else np.nan
    )

    # Összesített score
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
# ÚJ MODUL 4: OPTIMÁLIS TERHELÉSI ABLAK
# =========================================================

def compute_optimal_load_window(
    df: pd.DataFrame,
    load_col: str,
    tech_col: str = "Technika_index",
    res_col: str = "RES_plus",
    lag_weeks: int = 2,
) -> dict | None:
    """
    Lagged korreláció: heti terhelés vs következő lag_weeks hét Technika + RES átlaga.
    Meghatározza, melyik km/TSS tartományban volt a legjobb adaptáció.
    """
    if load_col not in df.columns or tech_col not in df.columns:
        return None
    w = df.dropna(subset=["Dátum"]).copy()
    w["week"] = w["Dátum"].dt.to_period("W").dt.start_time

    agg_spec: dict = {"load": (load_col, "sum")}
    if tech_col in w.columns:
        agg_spec["tech"] = (tech_col, "mean")
    if res_col in w.columns and w[res_col].notna().any():
        agg_spec["res"] = (res_col, "mean")

    weekly = w.groupby("week", as_index=False).agg(**agg_spec).sort_values("week")
    if len(weekly) < CFG["optload_min_weeks"]:
        return None

    if "tech" in weekly.columns:
        weekly["tech_future"] = weekly["tech"].shift(-lag_weeks).rolling(lag_weeks, min_periods=1).mean()
    if "res" in weekly.columns:
        weekly["res_future"] = weekly["res"].shift(-lag_weeks).rolling(lag_weeks, min_periods=1).mean()

    out_cols = [c for c in ["tech_future", "res_future"] if c in weekly.columns and weekly[c].notna().sum() >= 8]
    if not out_cols:
        return None

    weekly["outcome"] = weekly[out_cols].mean(axis=1)
    weekly = weekly.dropna(subset=["outcome", "load"])
    if len(weekly) < 8:
        return None

    weekly["load_bin"] = pd.qcut(weekly["load"], q=5, duplicates="drop")
    bin_means = (
        weekly.groupby("load_bin", observed=True)
        .agg(load_mid=("load", "median"), outcome_med=("outcome", "median"), count=("load", "count"))
        .reset_index()
        .sort_values("load_mid")
    )
    best_idx = bin_means["outcome_med"].idxmax()
    best_bin = bin_means.loc[best_idx]
    corr = float(weekly["load"].corr(weekly["outcome"]))

    lo = float(best_bin["load_bin"].left) if hasattr(best_bin["load_bin"], "left") else np.nan
    hi = float(best_bin["load_bin"].right) if hasattr(best_bin["load_bin"], "right") else np.nan

    return {
        "weekly": weekly,
        "bin_means": bin_means,
        "optimum_lo": lo,
        "optimum_hi": hi,
        "optimum_mid": float(best_bin["load_mid"]),
        "corr": corr,
        "lag_weeks": lag_weeks,
        "load_col": load_col,
        "n_weeks": len(weekly),
    }


def compute_intensity_ratio_effect(
    df: pd.DataFrame,
    run_type_col: str | None,
    tech_col: str = "Technika_index",
    lag_weeks: int = 3,
) -> dict | None:
    """
    Lagged korreláció: heti easy/tempo/race/interval arány vs következő
    lag_weeks hét Technika_index változása.
    """
    if run_type_col is None or run_type_col not in df.columns or tech_col not in df.columns:
        return None
    w = df.dropna(subset=["Dátum", run_type_col]).copy()
    w["week"] = w["Dátum"].dt.to_period("W").dt.start_time

    type_counts = (
        w.groupby(["week", run_type_col], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for t in ["easy", "tempo", "race", "interval"]:
        if t not in type_counts.columns:
            type_counts[t] = 0

    type_counts["total"] = type_counts[["easy", "tempo", "race", "interval"]].sum(axis=1)
    for t in ["easy", "tempo", "race", "interval"]:
        type_counts[f"{t}_pct"] = np.where(
            type_counts["total"] > 0, type_counts[t] / type_counts["total"] * 100, np.nan
        )

    tech_weekly = w.groupby("week")[tech_col].mean().reset_index()
    merged = type_counts.merge(tech_weekly, on="week", how="inner").sort_values("week")
    if len(merged) < 10:
        return None

    merged["tech_future"] = merged[tech_col].shift(-lag_weeks).rolling(lag_weeks, min_periods=1).mean()
    merged = merged.dropna(subset=["tech_future"])
    if len(merged) < 8:
        return None

    corrs = {}
    for t in ["easy", "tempo", "race", "interval"]:
        col = f"{t}_pct"
        if col in merged.columns and merged[col].notna().sum() >= 8:
            c = float(merged[col].corr(merged["tech_future"]))
            if not np.isnan(c):
                corrs[t] = c

    if not corrs:
        return None

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
        "corrs": corrs,
        "best_type": best_type,
        "opt_lo": opt_lo,
        "opt_hi": opt_hi,
        "merged": merged,
        "lag_weeks": lag_weeks,
    }


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
    # =========================================================
    # TECHNIKA_INDEX (slope-aware + speed bin) – vektorizált
    # Slope-korrekció: a terep hatását kivonjuk a nyers értékekből,
    # így az index terep-független "sík-ekvivalens" technikát tükröz.
    # =========================================================
    df["Technika_index"] = np.nan
    tech_base = df[mask_run & df["speed_mps"].notna()].copy()

    if len(tech_base) >= CFG["tech_min_total"]:
        # --- Slope-korrekció alkalmazása ---
        tech_base = apply_slope_correction(
            tech_base,
            cols=["vr_num", "gct_num", "vo_num", "cad_num", "stride_num"]
        )
        # A slope-corrected oszlopokat használjuk az index számításhoz
        _vr   = "vr_num_sc"   if "vr_num_sc"   in tech_base.columns else "vr_num"
        _gct  = "gct_num_sc"  if "gct_num_sc"  in tech_base.columns else "gct_num"
        _vo   = "vo_num_sc"   if "vo_num_sc"   in tech_base.columns else "vo_num"
        _cad  = "cad_num_sc"  if "cad_num_sc"  in tech_base.columns else "cad_num"
        _str  = "stride_num_sc" if "stride_num_sc" in tech_base.columns else "stride_num"

        tech_base["speed_bin"] = pd.qcut(
            tech_base["speed_mps"], q=CFG["speed_bins"], duplicates="drop"
        )
        for col in ["skill_vr", "skill_gct", "skill_vo", "skill_cad", "skill_stride"]:
            tech_base[col] = np.nan

        def _fill_tech_group(g: pd.DataFrame, target: pd.DataFrame):
            idx = g.index
            min_n = CFG["tech_min_group"]
            if g[_vr].notna().sum() >= min_n:
                target.loc[idx, "skill_vr"] = -robust_z(g[_vr], g[_vr])
            if g[_gct].notna().sum() >= min_n:
                target.loc[idx, "skill_gct"] = -robust_z(g[_gct], g[_gct])
            if g[_vo].notna().sum() >= min_n:
                target.loc[idx, "skill_vo"] = -robust_z(g[_vo], g[_vo])
            if g[_cad].notna().sum() >= min_n:
                target.loc[idx, "skill_cad"] = -np.abs(robust_z(g[_cad], g[_cad]))
            if g[_str].notna().sum() >= min_n:
                target.loc[idx, "skill_stride"] = robust_z(g[_str], g[_str])

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

    # --- Slope-korrekció: a HR és GCT/VR értékek terep-hatásának kivonása
    fat = apply_slope_correction(fat, cols=["gct_num", "vr_num", "cad_num", "hr_num", "pace_sec_km"])
    _fat_gct  = "gct_num_sc"      if "gct_num_sc"      in fat.columns else "gct_num"
    _fat_vr   = "vr_num_sc"       if "vr_num_sc"       in fat.columns else "vr_num"
    _fat_cad  = "cad_num_sc"      if "cad_num_sc"      in fat.columns else "cad_num"
    _fat_hr   = "hr_num_sc"       if "hr_num_sc"       in fat.columns else "hr_num"
    _fat_pace = "pace_sec_km_sc"  if "pace_sec_km_sc"  in fat.columns else "pace_sec_km"

    # HR/pace arány: slope-corrected értékekből számoljuk
    fat["hr_per_pace"] = np.where(
        fat[_fat_hr].notna() & fat[_fat_pace].notna() & (fat[_fat_pace] > 0),
        fat[_fat_hr] / fat[_fat_pace],
        np.nan,
    )

    if len(fat) >= CFG["fat_min_total"]:
        easy_base = fat[fat["Run_type"] == "easy"].copy() if "Run_type" in fat.columns else fat.copy()

        def _compute_fat_z_vectorized(
            fat_df: pd.DataFrame, easy_df: pd.DataFrame
        ) -> pd.DataFrame:
            """
            Vektorizált fatigue z-score számítás slope-corrected értékekből.
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

                for fat_col, src_col in [
                    ("fatigue_gct", _fat_gct),
                    ("fatigue_vr",  _fat_vr),
                    ("fatigue_hr",  "hr_per_pace"),
                ]:
                    if (
                        group[src_col].notna().sum() >= 5
                        and base[src_col].notna().sum() >= CFG["fat_min_baseline"]
                    ):
                        z = robust_z(group[src_col], base[src_col])
                        result.loc[idx, fat_col] = z.fillna(0).values

                if (
                    group[_fat_cad].notna().sum() >= 5
                    and base[_fat_cad].notna().sum() >= CFG["fat_min_baseline"]
                ):
                    z = robust_z(group[_fat_cad], base[_fat_cad])
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
        # --- Slope-korrekció a RES+ komponensekre
        eco = apply_slope_correction(
            eco, cols=["gct_num", "vr_num", "vo_num", "cad_num", "stride_num"]
        )
        _eco_gct    = "gct_num_sc"    if "gct_num_sc"    in eco.columns else "gct_num"
        _eco_vr     = "vr_num_sc"     if "vr_num_sc"     in eco.columns else "vr_num"
        _eco_vo     = "vo_num_sc"     if "vo_num_sc"     in eco.columns else "vo_num"
        _eco_cad    = "cad_num_sc"    if "cad_num_sc"    in eco.columns else "cad_num"
        _eco_stride = "stride_num_sc" if "stride_num_sc" in eco.columns else "stride_num"

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
            if g[_eco_gct].notna().sum() >= min_n:
                target.loc[idx, "eco_gct"] = -robust_z(g[_eco_gct], g[_eco_gct])
            if g[_eco_vr].notna().sum() >= min_n:
                target.loc[idx, "eco_vr"] = -robust_z(g[_eco_vr], g[_eco_vr])
            if g[_eco_vo].notna().sum() >= min_n:
                target.loc[idx, "eco_vo"] = -robust_z(g[_eco_vo], g[_eco_vo])
            if g[_eco_cad].notna().sum() >= min_n:
                target.loc[idx, "eco_cad"] = -np.abs(robust_z(g[_eco_cad], g[_eco_cad]))
            if g[_eco_stride].notna().sum() >= min_n:
                target.loc[idx, "eco_stride"] = robust_z(g[_eco_stride], g[_eco_stride])

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

    # =========================================================
    # TSS PROXY (a hrmax-ot itt nem ismerjük -> placeholder 185)
    # A valódi hrmax-ot a tab-ban alkalmazzuk, de 185-tel is elég jó közelítés
    # pipeline-on belüli számításhoz; a tab átszámolja ha user beállít mást.
    # =========================================================
    df["dur_sec"] = np.nan
    time_cands = [c for c in ["Idő", "Menetidő", "Eltelt idő"] if c in df.columns]
    if time_cands:
        df["dur_sec"] = df[time_cands[0]].apply(duration_to_seconds)

    df["TSS_proxy"] = compute_tss_proxy(df, hrmax=185)

    # =========================================================
    # ASZIMMETRIA
    # =========================================================
    df = compute_asymmetry(df)

    return df


# =========================================================
# ADATFORRÁS: STRAVA (auto) + GARMIN CSV (manuális / kiegészítő)
# =========================================================
st.sidebar.header("Adatforrás")

# --- Strava auto-szinkron
strava_df, strava_source = render_strava_connect_sidebar()

# --- Garmin CSV feltöltő (mindig látható, kiegészítő forrásként)
st.sidebar.divider()
st.sidebar.header("📂 Garmin CSV / XLSX")
st.sidebar.caption(
    "Running Dynamics adatokhoz (GCT, VO, VR) szükséges. "
    "Strava-val kombinálva a teljes elemzés elérhető."
)
uploaded = st.sidebar.file_uploader(
    "Garmin export (XLSX ajánlott)", type=["xlsx", "csv"]
)

# --- Forrás döntési logika
#   1. Ha van Strava ÉS Garmin: merge (Garmin adat bővíti a Strava-t Running Dynamics-szel)
#   2. Ha csak Strava: Strava alapú elemzés (Technika_index/Fatigue korlátozott)
#   3. Ha csak Garmin: eredeti működés
#   4. Egyik sem: welcome screen

def _merge_strava_garmin(strava_df: pd.DataFrame, garmin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Strava + Garmin adatok összeolvasztása dátum alapján.
    A Garmin adat elsőbbséget élvez – ha ugyanarra a napra van mindkettő,
    a Garmin Running Dynamics mezői (GCT, VO, VR, stride) feltöltik a Strava sort.
    """
    if strava_df.empty:
        return garmin_df
    if garmin_df.empty:
        return strava_df

    s = strava_df.copy()
    g = garmin_df.copy()

    # ── Timezone normalizálás: mindkettőt timezone-naive UTC-re hozzuk ──
    # Strava: lehet tz-aware (pl. 2024-01-15 08:30:00+01:00)
    # Garmin: tz-naive
    # Megoldás: mindkettőből eltávolítjuk a timezone infót (.dt.tz_localize(None))
    for _df in (s, g):
        if "Dátum" in _df.columns:
            dt = pd.to_datetime(_df["Dátum"], errors="coerce")
            if dt.dt.tz is not None:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
            _df["Dátum"] = dt

    # Nap-szintű egyeztetés
    s["_date"] = s["Dátum"].dt.date
    g["_date"] = g["Dátum"].dt.date

    # Garmin Running Dynamics oszlopok amik hiányoznak Stravából
    rd_cols = [c for c in ["vr_num", "gct_num", "vo_num", "stride_num",
                           "asym_gct_pct", "asym_stride_pct", "asym_power_pct", "Asymmetry_score"]
               if c in g.columns]

    if rd_cols:
        g_rd = g[["_date"] + rd_cols].copy()
        s = s.merge(g_rd, on="_date", how="left", suffixes=("", "_garmin"))
        for col in rd_cols:
            garmin_col = f"{col}_garmin"
            if garmin_col in s.columns:
                s[col] = s[col].combine_first(s[garmin_col])
                s.drop(columns=[garmin_col], inplace=True)

    s.drop(columns=["_date"], inplace=True, errors="ignore")

    # Garmin-only sorok (napok amik Stravában nem szerepelnek)
    s_dates = set(s["Dátum"].dt.date)
    g_only = g[~g["_date"].isin(s_dates)].drop(columns=["_date"], errors="ignore")
    merged = pd.concat([s, g_only], ignore_index=True).sort_values("Dátum")
    return merged


# --- Adatforrás összerakása
if strava_df is None and uploaded is None:
    st.title("🏃 Garmin Futás Dashboard")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🟠 Automatikus szinkron")
        st.info(
            "Csatlakozz Stravához a bal oldali sávban – "
            "minden futásod automatikusan betöltődik, "
            "CSV feltöltés nélkül."
        )
    with col_r:
        st.markdown("### 📂 Manuális feltöltés")
        st.info(
            "Vagy tölts fel egy Garmin Connect XLSX exportot. "
            "Ez szükséges a teljes biomechanikai elemzéshez "
            "(GCT, VO, VR, aszimmetria)."
        )
    st.stop()

with st.spinner("Adatok feldolgozása…"):
    if strava_df is not None and uploaded is not None:
        # Hibrid: Strava + Garmin merge
        file_bytes = uploaded.getvalue()
        garmin_df = full_pipeline(file_bytes, uploaded.name)
        # Strava df pipeline-on átfuttatjuk (TSS, aszimmetria számítások)
        strava_piped = full_pipeline(
            strava_df.to_csv(index=False).encode("utf-8"), "__strava__.csv"
        ) if False else strava_df  # pipeline csv-n át nem érdemes, direkt merge-elünk
        df = _merge_strava_garmin(strava_df, garmin_df)
        # Pipeline post-processzing ami a merge után kell (slope, temp_bin már megvan)
        if "TSS_proxy" not in df.columns:
            df["TSS_proxy"] = compute_tss_proxy(df, hrmax=185)
        if "Asymmetry_score" not in df.columns:
            df = compute_asymmetry(df)
        _data_source = "hybrid"

    elif strava_df is not None:
        # Csak Strava
        df = strava_df.copy()
        df["TSS_proxy"] = compute_tss_proxy(df, hrmax=185)
        df = compute_asymmetry(df)
        _data_source = "strava"

    else:
        # Csak Garmin CSV
        file_bytes = uploaded.getvalue()
        df = full_pipeline(file_bytes, uploaded.name)
        _data_source = "garmin"

# =========================================================
# ALAP SZŰRÉS + MEGJELENÍTÉSI ELŐKÉSZÍTÉS
# =========================================================
st.title("🏃 Garmin Futás Dashboard")

# Adatforrás badge
if _data_source == "strava":
    st.info(
        "🟠 **Strava alapú elemzés** – Running Dynamics adatok (GCT, VO, VR) hiányoznak. "
        "Technika_index és Fatigue_score nem számolható. "
        "Tölts fel Garmin XLSX exportot is a teljes elemzéshez.",
    )
elif _data_source == "hybrid":
    st.success(
        "✅ **Hibrid mód** – Strava automatikus szinkron + Garmin Running Dynamics. "
        "Minden elemzés elérhető.",
    )

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
hr_col = find_col(d, "hr_num", "Átlagos pulzusszám")
pace_col = find_col(d, "Átlagos tempó")
pwr_col = find_col(d, "power_avg_w", "Átl. teljesítmény")

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
tab_overview, tab_last, tab_warn, tab_ready, tab_pmc, tab_recovery, tab_asym, tab_strava, tab_strava_analysis, tab_data = st.tabs(
    ["📌 Áttekintés", "🔎 Utolsó futás", "🚦 Warning", "🏁 Readiness",
     "📈 PMC & Kockázat", "🔄 Recovery", "⚖️ Aszimmetria",
     "🟠 Strava adatok", "🟠 Strava elemzés", "📄 Adatok"]
)

# =========================================================
# TAB: ÁTTEKINTÉS
# =========================================================
with tab_overview:

    # =========================================================
    # 🏃 EASY RUN TARGET – mai állapot alapján
    # =========================================================
    st.subheader("🏃 Easy run target – mai állapot")

    _et = compute_easy_target(
        df=d,
        hrmax=hrmax,
        fatigue_col=fatigue_col,
        run_type_col=run_type_col,
        hr_col=hr_col,
        pace_col=pace_col,
        pwr_col=pwr_col,
        baseline_weeks=baseline_weeks,
        baseline_min_runs=baseline_min_runs,
    )

    if _et is None:
        st.info("Easy run target-hez legalább 5 easy futás szükséges HR adattal.")
    else:
        # --- Státusz banner
        rs = _et["readiness_status"]
        rt = _et["readiness_text"]
        if rs == "🔴":
            st.error(f"{rs} {rt}")
        elif rs == "🟠":
            st.warning(f"{rs} {rt}")
        elif rs == "🟡":
            st.warning(f"{rs} {rt}")
        else:
            st.success(f"{rs} {rt}")

        # --- Fő célsáv kártyák
        st.markdown("#### 🎯 Mai easy zóna")
        _c1, _c2, _c3 = st.columns(3)

        with _c1:
            st.markdown(
                f"""
                <div style="background:#1a1a2e;border-radius:14px;padding:18px 20px;text-align:center;">
                  <div style="font-size:0.85rem;color:#aaa;margin-bottom:4px;">❤️ Pulzus</div>
                  <div style="font-size:2rem;font-weight:700;color:#e74c3c;">
                    {_et['hr_lo']}–{_et['hr_hi']}
                  </div>
                  <div style="font-size:0.9rem;color:#aaa;">bpm</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with _c2:
            st.markdown(
                f"""
                <div style="background:#1a1a2e;border-radius:14px;padding:18px 20px;text-align:center;">
                  <div style="font-size:0.85rem;color:#aaa;margin-bottom:4px;">⏱️ Tempó</div>
                  <div style="font-size:2rem;font-weight:700;color:#3498db;">
                    {_et['pace_lo']}–{_et['pace_hi']}
                  </div>
                  <div style="font-size:0.9rem;color:#aaa;">min/km</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with _c3:
            if pd.notna(_et["pwr_lo"]) and pd.notna(_et["pwr_hi"]):
                _pwr_str = f"{int(_et['pwr_lo'])}–{int(_et['pwr_hi'])}"
                _pwr_unit = "W"
            else:
                _pwr_str = "—"
                _pwr_unit = "nincs power adat"
            st.markdown(
                f"""
                <div style="background:#1a1a2e;border-radius:14px;padding:18px 20px;text-align:center;">
                  <div style="font-size:0.85rem;color:#aaa;margin-bottom:4px;">⚡ Teljesítmény</div>
                  <div style="font-size:2rem;font-weight:700;color:#2ecc71;">
                    {_pwr_str}
                  </div>
                  <div style="font-size:0.9rem;color:#aaa;">{_pwr_unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Fáradtság-korrekció részletei
        if pd.notna(_et["current_fatigue"]) and _et["fatigue_penalty_factor"] > 0:
            with st.expander(
                f"📉 Fáradtság-korrekció alkalmazva "
                f"(Fatigue: {_et['current_fatigue']:.0f} → "
                f"−{_et['hr_penalty']:.1f} bpm / "
                f"+{_et['pace_penalty_sec']:.0f} sec/km / "
                f"−{_et['pwr_penalty']:.0f} W)"
            ):
                _col1, _col2 = st.columns(2)
                with _col1:
                    st.markdown("**Számítás alapja**")
                    st.markdown(f"- Jelenlegi Fatigue_score: **{_et['current_fatigue']:.0f}**")
                    st.markdown(f"- Korrekciós szorzó: **{_et['fatigue_penalty_factor']:.1f}×**")
                    st.markdown(f"- HR korrekció: **−{_et['hr_penalty']:.1f} bpm**")
                    st.markdown(f"- Tempó korrekció: **+{_et['pace_penalty_sec']:.0f} sec/km**")
                    if pd.notna(_et["pwr_penalty"]) and _et["pwr_penalty"] > 0:
                        st.markdown(f"- Power korrekció: **−{_et['pwr_penalty']:.0f} W**")
                with _col2:
                    st.markdown("**Forrásadatok**")
                    if pd.notna(_et["fatmax_hr"]):
                        st.markdown(f"- Fatmax HR: **{_et['fatmax_hr']:.0f} bpm** ({_et['fatmax_pace']} min/km)")
                    if pd.notna(_et["hr_med_easy"]):
                        st.markdown(f"- Easy medián HR: **{_et['hr_med_easy']:.0f} bpm**")
                    if pd.notna(_et["pace_med_easy"]) and _et["pace_med_easy"] != "—":
                        st.markdown(f"- Easy medián tempó: **{_et['pace_med_easy']} min/km**")
                    if pd.notna(_et["pwr_med_easy"]):
                        st.markdown(f"- Easy medián power: **{_et['pwr_med_easy']:.0f} W**")
        else:
            with st.expander("ℹ️ Számítás részletei"):
                if pd.notna(_et["fatmax_hr"]):
                    st.markdown(f"- Fatmax HR: **{_et['fatmax_hr']:.0f} bpm** ({_et['fatmax_pace']} min/km)")
                if pd.notna(_et["hr_med_easy"]):
                    st.markdown(f"- Easy medián HR: **{_et['hr_med_easy']:.0f} bpm**")
                if pd.notna(_et["pace_med_easy"]) and _et["pace_med_easy"] != "—":
                    st.markdown(f"- Easy medián tempó: **{_et['pace_med_easy']} min/km**")
                if pd.notna(_et["pwr_med_easy"]):
                    st.markdown(f"- Easy medián power: **{_et['pwr_med_easy']:.0f} W**")
                st.markdown(f"- Jelenlegi Fatigue: **{_et['current_fatigue']:.0f}** → nincs korrekció")

        # --- Historikus easy futások vs célsáv
        if not _et["easy_history"].empty and hr_col and hr_col in _et["easy_history"].columns:
            st.markdown("#### 📊 Legutóbbi easy futások HR vs. mai célsáv")
            _hist = _et["easy_history"].copy()
            _hist["HR"] = pd.to_numeric(_hist[hr_col], errors="coerce")
            _hist = _hist.dropna(subset=["Dátum", "HR"])

            if len(_hist) >= 3:
                fig_et = px.scatter(
                    _hist,
                    x="Dátum",
                    y="HR",
                    labels={"Dátum": "Dátum", "HR": "Átlag HR (bpm)"},
                    title="Easy futások átlag HR-je – célsávhoz viszonyítva",
                    color_discrete_sequence=["#3498db"],
                    opacity=0.75,
                )
                # Célsáv jelölése
                fig_et.add_hrect(
                    y0=_et["hr_lo"], y1=_et["hr_hi"],
                    fillcolor="green", opacity=0.12,
                    annotation_text=f"Mai célsáv ({_et['hr_lo']}–{_et['hr_hi']} bpm)",
                    annotation_position="top left",
                )
                # Rolling átlag
                _hist_sorted = _hist.sort_values("Dátum")
                if len(_hist_sorted) >= 6:
                    _hist_sorted["roll_hr"] = _hist_sorted["HR"].rolling(6, min_periods=3).mean()
                    fig_et.add_scatter(
                        x=_hist_sorted["Dátum"],
                        y=_hist_sorted["roll_hr"],
                        mode="lines",
                        name="6 futás átlag",
                        line=dict(color="white", dash="dot", width=2),
                    )
                st.plotly_chart(fig_et, use_container_width=True)

                # Pace trend ha van
                if "pace_sec_km" in _hist.columns and _hist["pace_sec_km"].notna().any():
                    _hist["Tempó (sec/km)"] = pd.to_numeric(_hist["pace_sec_km"], errors="coerce")
                    _pace_lo = pace_str_to_sec(_et["pace_lo"])
                    _pace_hi = pace_str_to_sec(_et["pace_hi"])
                    fig_pace_et = px.scatter(
                        _hist.dropna(subset=["Tempó (sec/km)"]),
                        x="Dátum",
                        y="Tempó (sec/km)",
                        labels={"Dátum": "Dátum", "Tempó (sec/km)": "Tempó (sec/km)"},
                        title="Easy futások tempója – célsávhoz viszonyítva",
                        color_discrete_sequence=["#e67e22"],
                        opacity=0.75,
                    )
                    if pd.notna(_pace_lo) and pd.notna(_pace_hi):
                        fig_pace_et.add_hrect(
                            y0=_pace_lo, y1=_pace_hi,
                            fillcolor="green", opacity=0.12,
                            annotation_text=f"Mai tempó célsáv ({_et['pace_lo']}–{_et['pace_hi']})",
                            annotation_position="top left",
                        )
                    # Y tengely: alacsonyabb sec/km = gyorsabb, fordított skálán jobb lenne
                    fig_pace_et.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_pace_et, use_container_width=True)

    st.divider()

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
                _safe_dropna(view, ["Technika_index"]),
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
            dd = _safe_dropna(view, ["Technika_index", fatigue_col]).copy()
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
            w_tt = _safe_dropna(d, ["Dátum", "Technika_index"]).copy()
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
        base = _safe_dropna(d, ["Dátum", "Technika_index"]).sort_values("Dátum")
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

            # ── Terep-profil és slope-korrekció összefoglaló ─────────
            _up   = float(last.get("up_m_per_km", 0)   or 0)
            _down = float(last.get("down_m_per_km", 0) or 0)
            _asc  = float(last.get("asc_m", 0)         or 0)
            _des  = float(last.get("des_m", 0)          or 0)
            _dist = float(last.get("dist_km", 0)        or 0)

            if _up > 0 or _down > 0:
                _sc_info = slope_correction_summary(_up, _down)
                _is_hilly = _up > 5 or _down > 5

                with st.expander(
                    f"⛰️ Terep-profil és slope-korrekció "
                    f"({'rolling' if _up < 15 else 'hilly'} – "
                    f"↑{_asc:.0f} m / ↓{_des:.0f} m, "
                    f"{_up:.1f} m/km emelkedés)"
                ):
                    _tc1, _tc2, _tc3 = st.columns(3)
                    _tc1.metric("Össz emelkedés", f"{_asc:.0f} m")
                    _tc2.metric("Össz süllyedés", f"{_des:.0f} m" if _des > 0 else "—")
                    _tc3.metric("Nettó szint", f"{_asc - _des:+.0f} m")

                    _tc4, _tc5, _tc6 = st.columns(3)
                    _tc4.metric("Emelkedés/km", f"{_up:.1f} m/km")
                    _tc5.metric("Süllyedés/km", f"{_down:.1f} m/km")
                    _tc6.metric("Távolság", f"{_dist:.2f} km")

                    if _is_hilly:
                        st.markdown("**Slope-korrekció hatása az indexekre** (kivont terep-hatás):")
                        _corr_rows = []
                        _labels = {
                            "vo_num":      ("Vertical Oscillation", "cm",    "↓ csökken emelkedőn → hamis javulás látszata"),
                            "gct_num":     ("Ground Contact Time",  "ms",    "↓ csökken emelkedőn → hamis javulás látszata"),
                            "vr_num":      ("Vertical Ratio",       "%",     "↓ kicsit csökken emelkedőn"),
                            "stride_num":  ("Lépéshossz",           "m",     "↑ nő emelkedőn → hamis javulás látszata"),
                            "cad_num":     ("Kadencia",             "spm",   "minimális változás"),
                            "hr_num":      ("Pulzus",               "bpm",   "↑ nő emelkedőn → hamis fáradtság látszata"),
                            "pace_sec_km": ("Tempó",                "sec/km","↓ csökken emelkedőn (gyorsabb sec/km)"),
                        }
                        for k, (lbl, unit, note) in _labels.items():
                            corr = _sc_info.get(k, 0)
                            if abs(corr) > 0.001:
                                _corr_rows.append({
                                    "Mutató": lbl,
                                    "Terep-hatás": f"{corr:+.2f} {unit}",
                                    "Megjegyzés": note,
                                    "Irány": "🟢 korrigálva" if abs(corr) > 0.5 else "🟡 kis hatás",
                                })
                        if _corr_rows:
                            st.dataframe(
                                pd.DataFrame(_corr_rows),
                                use_container_width=True, hide_index=True,
                            )

                        # VO külön kiemelés ha komoly emelkedő
                        if _up > 8:
                            _vo_corr = _sc_info.get("vo_num", 0)
                            _hr_corr = _sc_info.get("hr_num", 0)
                            st.info(
                                f"ℹ️ **Slope-korrekció összefoglalása** – {_up:.1f} m/km emelkedésnél:\n\n"
                                f"- **VO** terep-hatása: {_vo_corr:+.2f} cm – "
                                f"emelkedőn a VO tipikusan csökken, ezért ha az indexed mégis nőtt, "
                                f"az valódi technikai romlást jelent.\n"
                                f"- **HR** terep-hatása: {_hr_corr:+.2f} bpm – "
                                f"a Fatigue_score HR-komponensét ennyivel korrigáljuk lefelé.\n"
                                f"- **RES+** nem mérvadó ezen a terepen "
                                f"(sík baseline-hoz hasonlítja a dombos futást)."
                            )
                    else:
                        st.caption("Enyhe terep (< 5 m/km) – slope-korrekció minimális hatású.")

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

        warn_base = _safe_dropna(d, ["Dátum", "Technika_index"]).sort_values("Dátum")
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
        ready_base = _safe_dropna(d, ["Dátum", "Technika_index"]).sort_values("Dátum")
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
# TAB: PMC & SÉRÜLÉSKOCKÁZAT
# =========================================================
with tab_pmc:
    st.subheader("📈 Performance Management Chart – CTL / ATL / TSB")
    st.caption("CTL = fittség (42 napos EMA), ATL = fáradtság (7 napos EMA), TSB = forma (CTL − ATL).")

    # TSS újraszámítás a user HRmax-szal
    d_pmc = d.copy()
    if "TSS_proxy" in d_pmc.columns:
        d_pmc["TSS_proxy"] = compute_tss_proxy(d_pmc, hrmax)

    acwr_df = compute_acwr(d_pmc, load_col="TSS_proxy",
                            acute_days=CFG["acwr_acute_days"],
                            chronic_days=CFG["acwr_chronic_days"])

    if acwr_df.empty:
        st.info("Nincs elég TSS adat a PMC-hez (kell: Dátum + pulzus vagy tempó + táv).")
    else:
        pmc_df = compute_ctl_atl_tsb(acwr_df)

        # Aktuális értékek
        last_pmc = pmc_df.iloc[-1]
        acwr_now = float(last_pmc.get("acwr", np.nan))
        tsb_now = float(last_pmc.get("TSB", np.nan))
        ctl_now = float(last_pmc.get("CTL", np.nan))
        atl_now = float(last_pmc.get("ATL", np.nan))

        # Ramp rate (újra számolva itt)
        rr_pmc = np.nan
        if len(acwr_df) >= 35:
            last7 = acwr_df.tail(7)["load"].sum()
            prev28_mean = acwr_df.tail(35).head(28)["load"].mean()
            if prev28_mean > 0:
                rr_pmc = (last7 / 7 - prev28_mean) / prev28_mean * 100

        # Utolsó fatigue és aszimmetria
        fat_now = None
        if fatigue_col and d_pmc[fatigue_col].notna().any():
            fat_now = float(d_pmc.dropna(subset=[fatigue_col]).sort_values("Dátum").iloc[-1][fatigue_col])

        asym_now = None
        if "Asymmetry_score" in d_pmc.columns and d_pmc["Asymmetry_score"].notna().any():
            asym_now = float(d_pmc.dropna(subset=["Asymmetry_score"]).sort_values("Dátum").iloc[-1]["Asymmetry_score"])

        # Sérüléskockázat
        inj_score, inj_status, inj_expl = injury_risk_score(
            acwr_val=acwr_now,
            tsb_val=tsb_now,
            fatigue_val=fat_now,
            asym_val=asym_now,
            ramp_val=rr_pmc,
        )

        # --- KPI sor
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CTL (fittség)", f"{ctl_now:.1f}" if pd.notna(ctl_now) else "—")
        c2.metric("ATL (fáradtság)", f"{atl_now:.1f}" if pd.notna(atl_now) else "—")
        c3.metric("TSB (forma)", f"{tsb_now:+.1f}" if pd.notna(tsb_now) else "—",
                  delta=last_pmc.get("TSB_status", ""))
        c4.metric("ACWR", f"{acwr_now:.2f}" if pd.notna(acwr_now) else "—",
                  delta=last_pmc.get("acwr_status", ""))
        c5.metric("Sérüléskockázat", f"{inj_score:.0f}/100" if pd.notna(inj_score) else "—",
                  delta=inj_status)

        # Státusz doboz
        if inj_status.startswith("🔴"):
            st.error(f"**{inj_status}** sérüléskockázat")
        elif inj_status.startswith("🟠"):
            st.warning(f"**{inj_status}** sérüléskockázat")
        elif inj_status.startswith("🟡"):
            st.warning(f"**{inj_status}** sérüléskockázat")
        else:
            st.success(f"**{inj_status}** sérüléskockázat")

        with st.expander("🔍 Kockázati komponensek részletezése", expanded=True):
            for e in inj_expl:
                st.write(f"• {e}")

        st.divider()

        # --- PMC idősor (CTL / ATL / TSB)
        st.markdown("#### 📊 CTL / ATL / TSB idősor")
        pmc_plot = pmc_df.tail(180).copy()

        fig_pmc = px.line(
            pmc_plot, x="date", y=["CTL", "ATL"],
            labels={"date": "Dátum", "value": "TSS egység", "variable": "Mutató"},
            title="Fittség (CTL) vs Fáradtság (ATL)",
            color_discrete_map={"CTL": "#2ecc71", "ATL": "#e74c3c"},
        )
        st.plotly_chart(fig_pmc, use_container_width=True)

        fig_tsb = px.area(
            pmc_plot, x="date", y="TSB",
            title="Forma (TSB = CTL − ATL)",
            labels={"date": "Dátum", "TSB": "TSB (forma)"},
            color_discrete_sequence=["#3498db"],
        )
        fig_tsb.add_hline(y=CFG["tsb_green"], line_dash="dot", line_color="green",
                           annotation_text="Forma zóna")
        fig_tsb.add_hline(y=CFG["tsb_red"], line_dash="dot", line_color="red",
                           annotation_text="Kritikus zóna")
        fig_tsb.add_hline(y=0, line_color="gray", line_width=1)
        st.plotly_chart(fig_tsb, use_container_width=True)

        st.divider()

        # --- ACWR idősor
        st.markdown("#### ⚡ ACWR (Acute:Chronic Workload Ratio)")
        acwr_plot = acwr_df.tail(120).copy()
        fig_acwr = px.line(
            acwr_plot, x="date", y="acwr",
            title="ACWR idősor",
            labels={"date": "Dátum", "acwr": "ACWR"},
            color_discrete_sequence=["#e67e22"],
        )
        fig_acwr.add_hrect(y0=CFG["acwr_safe_lo"], y1=CFG["acwr_safe_hi"],
                            fillcolor="green", opacity=0.08, annotation_text="Optimális zóna")
        fig_acwr.add_hrect(y0=CFG["acwr_safe_hi"], y1=CFG["acwr_warn_hi"],
                            fillcolor="orange", opacity=0.08, annotation_text="Figyelmeztető")
        fig_acwr.add_hrect(y0=CFG["acwr_warn_hi"], y1=3.0,
                            fillcolor="red", opacity=0.08, annotation_text="Veszélyes")
        st.plotly_chart(fig_acwr, use_container_width=True)
        st.caption(
            "🟢 Optimális zóna: 0.8–1.3 | 🟠 Figyelmeztető: 1.3–1.5 | 🔴 Veszélyes: >1.5  "
            "(Gabbett & Hulin irodalmi küszöbök alapján)"
        )

        st.divider()

        # --- Optimális terhelési ablak
        st.markdown("#### 🎯 Optimális heti terhelési ablak (saját adataid alapján)")

        load_src = "TSS_proxy" if d_pmc["TSS_proxy"].notna().sum() >= 10 else "dist_km"
        load_label_opt = "Heti TSS" if load_src == "TSS_proxy" else "Heti km"

        opt = compute_optimal_load_window(
            d_pmc, load_col=load_src,
            lag_weeks=CFG["optload_lag_weeks"]
        )

        if opt is None:
            st.info(f"Nincs elég heti adat az optimum becsléshez (min {CFG['optload_min_weeks']} hét).")
        else:
            st.caption(
                f"Módszer: heti {load_label_opt} vs következő {opt['lag_weeks']} hét Technika + RES+ átlaga. "
                f"Korreláció: {opt['corr']:+.2f} | {opt['n_weeks']} hét alapján."
            )

            if pd.notna(opt["optimum_lo"]) and pd.notna(opt["optimum_hi"]):
                st.success(
                    f"🎯 Legjobb adaptáció **{opt['optimum_lo']:.0f}–{opt['optimum_hi']:.0f} {load_label_opt}** "
                    f"heti terhelésnél (medián: {opt['optimum_mid']:.0f})."
                )

            fig_opt = px.bar(
                opt["bin_means"],
                x="load_mid",
                y="outcome_med",
                title=f"Technika + RES+ outcome terhelési bin szerint ({load_label_opt})",
                labels={"load_mid": load_label_opt, "outcome_med": "Átlag Technika+RES (következő hetek)"},
                color="outcome_med",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig_opt, use_container_width=True)

            # Scatter: minden hét (trendvonal: numpy polyfit)
            fig_scatter_opt = px.scatter(
                opt["weekly"],
                x="load", y="outcome",
                labels={"load": load_label_opt, "outcome": "Jövőbeli Technika+RES átlag"},
                title=f"Heti terhelés vs jövőbeli adaptáció (+{opt['lag_weeks']} hét lag)",
            )
            _tmp = opt["weekly"].dropna(subset=["load", "outcome"])
            if len(_tmp) >= 3:
                _xf = _tmp["load"].to_numpy(dtype=float)
                _yf = _tmp["outcome"].to_numpy(dtype=float)
                _m, _b = np.polyfit(_xf, _yf, 1)
                _xs = np.linspace(_xf.min(), _xf.max(), 60)
                fig_scatter_opt.add_scatter(x=_xs, y=_m * _xs + _b,
                                             mode="lines", name="Trend",
                                             line=dict(color="red", dash="dash"))
            if pd.notna(opt["optimum_lo"]):
                fig_scatter_opt.add_vrect(
                    x0=opt["optimum_lo"], x1=opt["optimum_hi"],
                    fillcolor="green", opacity=0.1,
                    annotation_text="Optimum tartomány",
                )
            st.plotly_chart(fig_scatter_opt, use_container_width=True)

        st.divider()

        # --- Intenzitás-arány hatás
        st.markdown("#### 🧩 Intenzitás-arány hatása az adaptációra (saját adataid)")
        ir = compute_intensity_ratio_effect(
            d_pmc, run_type_col=run_type_col,
            lag_weeks=CFG["intratio_lag_weeks"]
        )

        if ir is None:
            st.info("Nincs elég adat az intenzitás-arány elemzéshez.")
        else:
            # Korreláció bar chart
            corr_df = pd.DataFrame({
                "Típus": list(ir["corrs"].keys()),
                "Korreláció": list(ir["corrs"].values()),
            }).sort_values("Korreláció", ascending=False)
            corr_df["Típus_hu"] = corr_df["Típus"].apply(run_type_hu)

            fig_corr = px.bar(
                corr_df, x="Típus_hu", y="Korreláció",
                color="Korreláció",
                color_continuous_scale="RdYlGn",
                title=f"Heti edzéstípus-arány korrelációja a jövőbeli Technikával (+{ir['lag_weeks']} hét)",
                labels={"Típus_hu": "Edzés típusa", "Korreláció": "Pearson r"},
            )
            fig_corr.add_hline(y=0, line_color="gray")
            st.plotly_chart(fig_corr, use_container_width=True)

            best_hu = run_type_hu(ir["best_type"])
            if pd.notna(ir["opt_lo"]) and pd.notna(ir["opt_hi"]):
                st.success(
                    f"📐 A legjobb **{best_hu}** arány a következő hetekre: "
                    f"**{ir['opt_lo']:.0f}–{ir['opt_hi']:.0f}%** (az összes futásból)."
                )

            # Idősoros kép: easy % vs future tech
            best_col = f"{ir['best_type']}_pct"
            if best_col in ir["merged"].columns:
                _ir_df = ir["merged"].dropna(subset=[best_col, "tech_future"])
                fig_ir = px.scatter(
                    _ir_df,
                    x=best_col, y="tech_future",
                    labels={best_col: f"{best_hu} arány (%)", "tech_future": f"Technika +{ir['lag_weeks']} hét"},
                    title=f"{best_hu} arány vs jövőbeli Technika",
                )
                if len(_ir_df) >= 3:
                    _xf = _ir_df[best_col].to_numpy(dtype=float)
                    _yf = _ir_df["tech_future"].to_numpy(dtype=float)
                    _m, _b = np.polyfit(_xf, _yf, 1)
                    _xs = np.linspace(_xf.min(), _xf.max(), 60)
                    fig_ir.add_scatter(x=_xs, y=_m * _xs + _b,
                                       mode="lines", name="Trend",
                                       line=dict(color="red", dash="dash"))
                if pd.notna(ir["opt_lo"]):
                    fig_ir.add_vrect(
                        x0=ir["opt_lo"], x1=ir["opt_hi"],
                        fillcolor="green", opacity=0.1,
                        annotation_text="Optimum",
                    )
                st.plotly_chart(fig_ir, use_container_width=True)

        with st.expander("📋 PMC napi adatok (utolsó 60 nap)"):
            st.dataframe(pmc_df.tail(60)[["date", "load", "acute", "chronic", "acwr", "acwr_status", "CTL", "ATL", "TSB", "TSB_status"]],
                         use_container_width=True, hide_index=True)


# =========================================================
# TAB: RECOVERY TIME MODELL
# =========================================================
with tab_recovery:
    st.subheader("🔄 Recovery Time Modell")
    st.caption(
        "Megmutatja, hogy egy adott Fatigue_score szintű futás után "
        "saját adataid alapján hány napra volt szükség a Technika_index "
        "visszaállásához. Így becsülhető, mikor \"vagy készen\" az intenzív terhelésre."
    )

    if fatigue_col is None or "Technika_index" not in d.columns:
        st.info("Recovery modellhez kell Fatigue_score és Technika_index.")
    else:
        rec = compute_recovery_model(
            d,
            fatigue_col=fatigue_col,
            run_type_col=run_type_col,
            lookback_weeks=CFG["recovery_lookback_weeks"],
            min_events=CFG["recovery_min_events"],
        )

        if rec is None:
            st.info(
                "Nincs elég adat a recovery modellhez. "
                f"Kell: legalább ~{CFG['recovery_min_events']} magas-fatigue esemény + "
                "utánuk legalább 2 easy futás 30 napon belül."
            )
        else:
            # --- KPI sor
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Medián recovery", f"{rec['median_recovery']:.0f} nap")
            c2.metric("Elemzett események", f"{rec['n_events']}")
            c3.metric("Jelenlegi Fatigue", f"{rec['current_fat']:.0f}" if rec["current_fat"] else "—")
            c4.metric(
                "Becsült recovery",
                f"{rec['predicted_days']} nap" if rec["predicted_days"] is not None else "—",
                delta="holnaptól számítva" if rec["predicted_days"] else None,
            )

            if rec["predicted_days"] is not None:
                p = rec["predicted_days"]
                if p <= 1:
                    st.success(f"🟢 **Valószínűleg felépültél** – becsült recovery: {p} nap.")
                elif p <= 3:
                    st.warning(f"🟠 **Még {p} nap ajánlott** könnyű edzés / pihenő a teli intenzitás előtt.")
                else:
                    st.error(f"🔴 **Becsült recovery: {p} nap** – ne emelj terhelést még.")

            st.divider()

            # --- Scatter: fatigue vs recovery_days + trendvonal (numpy polyfit)
            fig_rec = px.scatter(
                rec["df"],
                x="fatigue",
                y="recovery_days",
                labels={"fatigue": "Fatigue_score az eseménynél", "recovery_days": "Recovery napok"},
                title="Fatigue_score vs Recovery idő (saját adatok alapján)",
            )
            if len(rec["df"]) >= 3:
                _xf = rec["df"]["fatigue"].to_numpy(dtype=float)
                _yf = rec["df"]["recovery_days"].to_numpy(dtype=float)
                _m, _b = np.polyfit(_xf, _yf, 1)
                _xs = np.linspace(_xf.min(), _xf.max(), 60)
                fig_rec.add_scatter(x=_xs, y=_m * _xs + _b,
                                    mode="lines", name="Trend",
                                    line=dict(color="orange", dash="dash"))

            # Aktuális fatigue megjelölése
            if rec["current_fat"] is not None and rec["predicted_days"] is not None:
                fig_rec.add_scatter(
                    x=[rec["current_fat"]],
                    y=[rec["predicted_days"]],
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="star"),
                    name="Jelenlegi becslés",
                )

            st.plotly_chart(fig_rec, use_container_width=True)

            # --- Hisztogram: recovery napok eloszlása
            fig_hist = px.histogram(
                rec["df"],
                x="recovery_days",
                nbins=12,
                title="Recovery napok eloszlása",
                labels={"recovery_days": "Recovery napok"},
                color_discrete_sequence=["#3498db"],
            )
            fig_hist.add_vline(
                x=rec["median_recovery"],
                line_dash="dash", line_color="orange",
                annotation_text=f"Medián: {rec['median_recovery']:.0f} nap",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- Recovery görbék: technika alakulása esemény után
            st.markdown("#### 📉 Technika alakulása magas-Fatigue esemény után")
            base_rec = _safe_dropna(d, ["Dátum", "Technika_index", fatigue_col]).sort_values("Dátum").copy()
            events_rec = base_rec[base_rec[fatigue_col] > 65].copy()
            if len(events_rec) < 3:
                thr_r = np.nanpercentile(base_rec[fatigue_col], 75)
                events_rec = base_rec[base_rec[fatigue_col] >= thr_r].copy()

            if run_type_col and run_type_col in base_rec.columns:
                easy_rec = base_rec[base_rec[run_type_col] == "easy"].copy()
            else:
                easy_rec = base_rec.copy()

            curves = []
            for ev_idx, (_, ev) in enumerate(events_rec.tail(8).iterrows()):
                ev_date = ev["Dátum"]
                post = easy_rec[
                    (easy_rec["Dátum"] > ev_date) &
                    (easy_rec["Dátum"] <= ev_date + pd.Timedelta(days=21))
                ].sort_values("Dátum").copy()
                if len(post) < 2:
                    continue
                post["days_after"] = (post["Dátum"] - ev_date).dt.days
                post["event_label"] = ev_date.strftime("%Y-%m-%d")
                post["fat_at_event"] = round(float(ev[fatigue_col]), 0)
                curves.append(post[["days_after", "Technika_index", "event_label", "fat_at_event"]])

            if curves:
                curve_df = pd.concat(curves, ignore_index=True)
                fig_curves = px.line(
                    curve_df,
                    x="days_after",
                    y="Technika_index",
                    color="event_label",
                    hover_data=["fat_at_event"],
                    labels={"days_after": "Napok az esemény után", "Technika_index": "Technika_index", "event_label": "Esemény dátuma"},
                    title="Technika visszaépülési görbék (legutóbbi 8 magas-Fatigue esemény)",
                )
                st.plotly_chart(fig_curves, use_container_width=True)
                st.caption("Minden vonal egy esemény utáni easy futássorozatot mutat.")
            else:
                st.info("Nincs elég post-event easy futás a görbék rajzolásához.")

            with st.expander("📋 Recovery esemény adatok"):
                st.dataframe(rec["df"].sort_values("fatigue", ascending=False),
                             use_container_width=True, hide_index=True)


# =========================================================
# TAB: ASZIMMETRIA
# =========================================================
with tab_asym:
    st.subheader("⚖️ Futási aszimmetria elemzés")
    st.caption(
        "Bal/jobb oldali különbségek (GCT, lépéshossz, power balance). "
        ">3% figyelmeztető, >5% sérülésprediktív határ (irodalmi konszenzus)."
    )

    asym_available = any(
        c in d.columns and d[c].notna().any()
        for c in ["asym_gct_pct", "asym_stride_pct", "asym_power_pct", "Asymmetry_score"]
    )

    if not asym_available:
        st.info(
            "Nincs bal/jobb aszimmetria adat a feltöltött fájlban. "
            "Ehhez szükséges: **Bal GCT / Jobb GCT**, **Bal lépéshossz / Jobb lépéshossz**, "
            "vagy **Egyensúly** (power balance %) Garmin oszlopok."
        )
        st.markdown(
            "**Hogyan exportálhatod?** Garmin Connect → Tevékenységek → "
            "Export CSV → győződj meg, hogy a Running Dynamics mezők be vannak kapcsolva "
            "az eszköz beállításaiban (Running Dynamics Pod vagy kompatibilis óra szükséges)."
        )
    else:
        asym_base = d.dropna(subset=["Dátum"]).sort_values("Dátum").copy()

        # --- Aktuális aszimmetria (utolsó futás)
        last_asym = asym_base.dropna(subset=["Asymmetry_score"]).iloc[-1] if asym_base["Asymmetry_score"].notna().any() else None

        if last_asym is not None:
            asym_val = float(last_asym["Asymmetry_score"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Összesített aszimmetria", f"{asym_val:.1f}%")
            c2.metric("GCT aszimmetria", f"{float(last_asym.get('asym_gct_pct', np.nan)):.1f}%" if pd.notna(last_asym.get("asym_gct_pct")) else "—")
            c3.metric("Lépéshossz aszimmetria", f"{float(last_asym.get('asym_stride_pct', np.nan)):.1f}%" if pd.notna(last_asym.get("asym_stride_pct")) else "—")
            c4.metric("Power balance eltérés", f"{float(last_asym.get('asym_power_pct', np.nan)):.1f}%" if pd.notna(last_asym.get("asym_power_pct")) else "—")

            if asym_val >= CFG["asym_red_pct"]:
                st.error(f"🔴 Magas aszimmetria ({asym_val:.1f}%) – sérülésprediktív jel. Javasolt: erősítő edzés + fizioterápiás konzultáció.")
            elif asym_val >= CFG["asym_warn_pct"]:
                st.warning(f"🟠 Figyelemre méltó aszimmetria ({asym_val:.1f}%) – kövesd figyelemmel az alakulást.")
            else:
                st.success(f"🟢 Jó szimmetria ({asym_val:.1f}%) – nincs aggodalomra ok.")

        st.divider()

        # --- Aszimmetria idősor
        asym_cols_present = [c for c in ["asym_gct_pct", "asym_stride_pct", "asym_power_pct", "Asymmetry_score"]
                             if c in asym_base.columns and asym_base[c].notna().any()]

        if asym_cols_present:
            asym_labels = {
                "asym_gct_pct": "GCT aszimmetria (%)",
                "asym_stride_pct": "Lépéshossz aszimmetria (%)",
                "asym_power_pct": "Power balance eltérés (%)",
                "Asymmetry_score": "Összesített aszimmetria (%)",
            }
            sel_asym = st.selectbox(
                "Melyik aszimmetria mutatót nézzük?",
                options=asym_cols_present,
                format_func=lambda x: asym_labels.get(x, x),
                key="asym_sel",
            )

            plot_asym = asym_base.dropna(subset=[sel_asym]).copy()
            color_col = run_type_col if run_type_col and run_type_col in plot_asym.columns else None

            fig_asym = px.scatter(
                plot_asym,
                x="Dátum",
                y=sel_asym,
                color="Edzés típusa" if "Edzés típusa" in plot_asym.columns else color_col,
                hover_data=[c for c in ["Cím", "dist_km", "Átlagos tempó"] if c in plot_asym.columns],
                labels={"Dátum": "Dátum", sel_asym: asym_labels.get(sel_asym, sel_asym)},
                title=f"{asym_labels.get(sel_asym, sel_asym)} idősor",
                opacity=0.7,
            )
            # Rolling átlag
            roll_asym = plot_asym[["Dátum", sel_asym]].sort_values("Dátum").copy()
            if len(roll_asym) >= 10:
                roll_asym["roll"] = roll_asym[sel_asym].rolling(10, min_periods=5).mean()
                for tr in px.line(roll_asym, x="Dátum", y="roll", color_discrete_sequence=["black"]).data:
                    tr.name = "10 futás átlag"
                    tr.showlegend = True
                    fig_asym.add_trace(tr)

            fig_asym.add_hline(y=CFG["asym_warn_pct"], line_dash="dot", line_color="orange",
                                annotation_text=f"Figyelmeztetés ({CFG['asym_warn_pct']}%)")
            fig_asym.add_hline(y=CFG["asym_red_pct"], line_dash="dot", line_color="red",
                                annotation_text=f"Kockázati küszöb ({CFG['asym_red_pct']}%)")
            st.plotly_chart(fig_asym, use_container_width=True)

            # --- Aszimmetria vs Fatigue kapcsolat
            st.divider()
            st.markdown("#### 🔗 Aszimmetria vs Fatigue_score")
            if fatigue_col and fatigue_col in asym_base.columns:
                af_df = asym_base.dropna(subset=[sel_asym, fatigue_col]).copy()
                if len(af_df) >= 10:
                    fig_af = px.scatter(
                        af_df,
                        x=fatigue_col,
                        y=sel_asym,
                        color="Edzés típusa" if "Edzés típusa" in af_df.columns else None,
                        labels={fatigue_col: "Fatigue_score", sel_asym: asym_labels.get(sel_asym, sel_asym)},
                        title="Összefügg-e a fáradtság az aszimmetriával?",
                        opacity=0.7,
                    )
                    _xf = af_df[fatigue_col].to_numpy(dtype=float)
                    _yf = af_df[sel_asym].to_numpy(dtype=float)
                    _valid = ~(np.isnan(_xf) | np.isnan(_yf))
                    if _valid.sum() >= 3:
                        _m, _b = np.polyfit(_xf[_valid], _yf[_valid], 1)
                        _xs = np.linspace(_xf[_valid].min(), _xf[_valid].max(), 60)
                        fig_af.add_scatter(x=_xs, y=_m * _xs + _b,
                                           mode="lines", name="Trend",
                                           line=dict(color="red", dash="dash"))
                    corr_af = float(af_df[fatigue_col].corr(af_df[sel_asym]))
                    st.plotly_chart(fig_af, use_container_width=True)
                    if abs(corr_af) >= 0.4:
                        st.info(f"📊 Korreláció (Fatigue ↔ Aszimmetria): **{corr_af:+.2f}** – látható összefüggés.")
                    else:
                        st.caption(f"Korreláció (Fatigue ↔ Aszimmetria): {corr_af:+.2f} – nincs erős kapcsolat.")
                else:
                    st.info("Kevés átfedő adat a Fatigue–Aszimmetria elemzéshez.")
            else:
                st.info("Fatigue_score hiányában nem számolható a korreláció.")

            # --- Aszimmetria statisztika
            st.divider()
            st.markdown("#### 📊 Aszimmetria statisztika")
            stat_cols = [c for c in ["asym_gct_pct", "asym_stride_pct", "asym_power_pct", "Asymmetry_score"]
                         if c in asym_base.columns and asym_base[c].notna().any()]
            if stat_cols:
                stat_rows = []
                for sc in stat_cols:
                    series = asym_base[sc].dropna()
                    stat_rows.append({
                        "Mutató": asym_labels.get(sc, sc),
                        "Medián (%)": f"{series.median():.2f}",
                        "Átlag (%)": f"{series.mean():.2f}",
                        "Max (%)": f"{series.max():.2f}",
                        "Futások >3%": int((series > 3).sum()),
                        "Futások >5%": int((series > 5).sum()),
                    })
                st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

        with st.expander("📋 Aszimmetria adatok (szűrt nézet)"):
            asym_show_cols = ["Dátum"] + [c for c in ["asym_gct_pct", "asym_stride_pct", "asym_power_pct", "Asymmetry_score"] if c in view.columns]
            if "Edzés típusa" in view.columns:
                asym_show_cols.insert(1, "Edzés típusa")
            st.dataframe(
                view.dropna(subset=[c for c in asym_show_cols if c in view.columns][2:3])
                    .sort_values("Dátum", ascending=False)[asym_show_cols],
                use_container_width=True, hide_index=True,
            )


# =========================================================
# TAB: STRAVA ADATOK
# =========================================================
with tab_strava:
    st.subheader("🟠 Strava adatok – részletes nézet")

    if _data_source == "garmin":
        st.info(
            "Ez a tab Strava csatlakozás esetén aktív. "
            "Jelenleg Garmin CSV módban fut az app – "
            "csatlakozz Stravához a bal oldali sávban."
        )
    else:
        # ── Utolsó futás nyers Strava JSON ──────────────────────────
        st.markdown("### 🔍 Utolsó futás – nyers Strava mezők")
        st.caption(
            "Ez mutatja pontosan mit küld át a Strava API egy futásnál. "
            "Látható melyik mező van meg és melyik hiányzik."
        )

        access_token = st.session_state.get("strava_access_token")
        if access_token:
            # Legutóbbi 1 futás letöltése részletesen
            last_act = _strava_get(
                "/athlete/activities",
                access_token,
                params={"per_page": 5, "page": 1},
            )
            run_acts = [a for a in (last_act or [])
                        if a.get("sport_type", a.get("type", "")) in
                        ("Run", "TrailRun", "VirtualRun", "Race", "Workout")]

            if run_acts:
                # Futás választó
                options = {
                    f"{a.get('start_date_local','')[:10]}  –  {a.get('name','?')}  "
                    f"({a.get('distance',0)/1000:.1f} km)": a
                    for a in run_acts
                }
                chosen_label = st.selectbox(
                    "Melyik futást nézzük?",
                    options=list(options.keys()),
                    key="strava_debug_sel",
                )
                act = options[chosen_label]

                # ── 1. Összefoglaló kártyák ──────────────────────────
                st.markdown("#### 📊 Összefoglaló")
                _c = st.columns(4)
                _c[0].metric("Távolság", f"{act.get('distance',0)/1000:.2f} km")
                _c[1].metric("Idő", f"{int(act.get('moving_time',0)//60)} perc")
                hr_v = act.get("average_heartrate")
                _c[2].metric("Átlag HR", f"{hr_v:.0f} bpm" if hr_v else "—")
                cad_v = act.get("average_cadence")
                _c[3].metric("Kadencia", f"{cad_v*2:.0f} spm" if cad_v else "—")

                _c2 = st.columns(4)
                spd = act.get("average_speed", 0)
                pace_s = 1000/spd if spd > 0 else None
                _c2[0].metric("Tempó", sec_to_pace_str(pace_s) + " /km" if pace_s else "—")
                _c2[1].metric("Emelkedés", f"{act.get('total_elevation_gain',0):.0f} m")
                pwr_v = act.get("average_watts")
                _c2[2].metric("Átlag power", f"{pwr_v:.0f} W" if pwr_v else "—")
                suf = act.get("suffer_score")
                _c2[3].metric("Suffer score", f"{suf}" if suf else "—")

                st.divider()

                # ── 2. Mezők: Van / Nincs táblázat ──────────────────
                st.markdown("#### ✅ Strava mezők – mi érkezett meg?")

                STRAVA_FIELDS = {
                    # Alapadatok
                    "name":                     ("Cím / aktivitás neve", "alap"),
                    "start_date_local":         ("Dátum (helyi idő)", "alap"),
                    "distance":                 ("Távolság (m)", "alap"),
                    "moving_time":              ("Mozgási idő (s)", "alap"),
                    "elapsed_time":             ("Eltelt idő (s)", "alap"),
                    "total_elevation_gain":     ("Emelkedés (m)", "alap"),
                    "sport_type":               ("Sport típusa", "alap"),
                    "average_speed":            ("Átlag sebesség (m/s)", "alap"),
                    "max_speed":                ("Max sebesség (m/s)", "alap"),
                    # Szív és erőfeszítés
                    "average_heartrate":        ("Átlag pulzus (bpm)", "szív"),
                    "max_heartrate":            ("Max pulzus (bpm)", "szív"),
                    "suffer_score":             ("Suffer score", "szív"),
                    "perceived_exertion":       ("Érzett erőfeszítés (1-10)", "szív"),
                    # Futótechnika
                    "average_cadence":          ("Átlag kadencia (jobb láb/perc)", "technika"),
                    "average_watts":            ("Átlag teljesítmény (W)", "technika"),
                    "max_watts":                ("Max teljesítmény (W)", "technika"),
                    "weighted_average_watts":   ("Súlyozott átlag W (NP)", "technika"),
                    "device_watts":             ("Valódi power mérő?", "technika"),
                    # Elhelyezkedés
                    "start_latlng":             ("Indulási koordináta", "helyszín"),
                    "end_latlng":               ("Érkezési koordináta", "helyszín"),
                    "map":                      ("GPS térkép (polyline)", "helyszín"),
                    # Egyéb
                    "average_temp":             ("Hőmérséklet (°C)", "egyéb"),
                    "calories":                 ("Kalória", "egyéb"),
                    "kudos_count":              ("Kudos szám", "egyéb"),
                    "achievement_count":        ("Teljesítmények száma", "egyéb"),
                    "pr_count":                 ("Személyes rekordok száma", "egyéb"),
                    "gear_id":                  ("Cipő / eszköz ID", "egyéb"),
                    "trainer":                  ("Futópad?", "egyéb"),
                    "commute":                  ("Ingázás?", "egyéb"),
                    # Garminban van, Stravában NINCS
                    "vertical_oscillation":     ("⛔ Függőleges oszcilláció (VO)", "hiányzik"),
                    "ground_contact_time":      ("⛔ Talajérintési idő (GCT)", "hiányzik"),
                    "vertical_ratio":           ("⛔ Függőleges arány (VR)", "hiányzik"),
                    "stride_length":            ("⛔ Lépéshossz", "hiányzik"),
                    "left_right_balance":       ("⛔ Bal/jobb egyensúly", "hiányzik"),
                }

                rows_debug = []
                for field, (label, category) in STRAVA_FIELDS.items():
                    val = act.get(field)
                    has_val = val is not None and val != "" and val != []

                    if category == "hiányzik":
                        status_icon = "❌"
                        display_val = "Nincs Strava API-ban"
                    elif has_val:
                        status_icon = "✅"
                        # Formázott érték
                        if field == "distance":
                            display_val = f"{float(val)/1000:.3f} km"
                        elif field == "average_speed":
                            pace_sec = 1000 / float(val) if float(val) > 0 else None
                            display_val = f"{sec_to_pace_str(pace_sec)} /km ({float(val):.2f} m/s)"
                        elif field == "moving_time" or field == "elapsed_time":
                            m, s = divmod(int(val), 60)
                            h, m = divmod(m, 60)
                            display_val = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
                        elif field == "average_cadence":
                            display_val = f"{float(val)*2:.0f} spm (Strava: {float(val):.1f} × 2)"
                        elif field == "map":
                            display_val = "✓ GPS útvonal megvan (polyline)"
                        elif field == "start_latlng" or field == "end_latlng":
                            display_val = f"{val[0]:.5f}, {val[1]:.5f}" if val else "—"
                        else:
                            display_val = str(val)
                    else:
                        status_icon = "⚠️"
                        display_val = "Nincs adat (None)"

                    rows_debug.append({
                        "": status_icon,
                        "Strava mező": field,
                        "Magyar név": label,
                        "Kategória": category,
                        "Érték": display_val,
                    })

                df_debug = pd.DataFrame(rows_debug)

                # Kategória szűrő
                cats = ["összes"] + sorted(df_debug["Kategória"].unique().tolist())
                sel_cat = st.radio(
                    "Kategória szűrő",
                    options=cats,
                    horizontal=True,
                    key="strava_cat_filter",
                )
                if sel_cat != "összes":
                    df_debug = df_debug[df_debug["Kategória"] == sel_cat]

                # Csak megvan / hiányzik szűrő
                col_f1, col_f2 = st.columns(2)
                show_missing = col_f1.checkbox("⚠️ Hiányzók megjelenítése", value=True)
                show_garmin_only = col_f2.checkbox("❌ Garmin-only mezők", value=True)
                if not show_missing:
                    df_debug = df_debug[df_debug[""] != "⚠️"]
                if not show_garmin_only:
                    df_debug = df_debug[df_debug[""] != "❌"]

                st.dataframe(
                    df_debug,
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                    column_config={
                        "": st.column_config.TextColumn(width="small"),
                        "Strava mező": st.column_config.TextColumn(width="medium"),
                        "Magyar név": st.column_config.TextColumn(width="large"),
                        "Kategória": st.column_config.TextColumn(width="small"),
                        "Érték": st.column_config.TextColumn(width="large"),
                    }
                )

                # ── 3. Összefoglaló számok ───────────────────────────
                n_ok   = (df_debug[""] == "✅").sum()
                n_warn = (df_debug[""] == "⚠️").sum()
                n_miss = (df_debug[""] == "❌").sum()
                st.caption(
                    f"✅ {n_ok} mező megvan  |  "
                    f"⚠️ {n_warn} mező hiányzik (de lekérhető lenne)  |  "
                    f"❌ {n_miss} mező Garmin-only (Strava API-ban nincs)"
                )

                st.divider()

                # ── 4. Teljes nyers JSON ─────────────────────────────
                with st.expander("🔩 Teljes nyers Strava JSON (fejlesztői nézet)"):
                    # Érzékeny mezők maszkolása
                    safe_act = {k: v for k, v in act.items()
                                if k not in ("map",)}  # polyline nem kell
                    if "map" in act:
                        safe_act["map"] = {"summary_polyline": "...(elrejtve)..."}
                    st.json(safe_act)

                st.divider()

                # ── 5. Mit tud és mit nem a dashboard Strava módban ──
                st.markdown("#### 📋 Dashboard képességek Strava vs Garmin módban")
                capability_data = {
                    "Funkció": [
                        "ACWR / TSS / CTL / ATL / TSB",
                        "Easy Run Target (HR + tempó)",
                        "Easy Run Target (power)",
                        "Ramp rate & heti terhelés",
                        "Fatmax becslés",
                        "Aerobic decoupling",
                        "Recovery time modell",
                        "Technika_index",
                        "Fatigue_score",
                        "RES+ (Running Economy Score)",
                        "Aszimmetria elemzés",
                        "Slope-aware elemzés",
                        "Hőmérséklet-korrekció",
                    ],
                    "🟠 Strava": [
                        "✅ Teljes",
                        "✅ HR alapon",
                        "✅ ha van power mérő",
                        "✅ Teljes",
                        "✅ HR + tempó alapon",
                        "✅ HR + tempó alapon",
                        "⚠️ Korlátozott (Fatigue hiány)",
                        "❌ GCT/VO/VR hiányzik",
                        "❌ GCT/VO/VR hiányzik",
                        "⚠️ Részleges (HR/power alapon)",
                        "❌ Bal/jobb adat hiányzik",
                        "⚠️ Csak emelkedés alapján",
                        "✅ ha van hőmérséklet adat",
                    ],
                    "📂 Garmin CSV": [
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ ha Running Dynamics van",
                        "✅ Teljes",
                        "✅ Teljes",
                    ],
                    "🔀 Hibrid (mindkettő)": [
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                        "✅ Teljes",
                    ],
                }
                st.dataframe(
                    pd.DataFrame(capability_data),
                    use_container_width=True,
                    hide_index=True,
                    height=490,
                )

            else:
                st.info("Nem találtam futós aktivitást az utolsó 5 aktivitás között.")
        else:
            st.warning("Strava access token nem elérhető – lehet hogy lejárt a session. Frissítsd az oldalt.")


# =========================================================
# TAB: STRAVA ELEMZÉS
# =========================================================
with tab_strava_analysis:
    st.subheader("🟠 Strava elemzés – csak Strava adatok alapján")
    st.caption("Távolság, tempó, HR, kadencia, emelkedés, suffer score – mind a Strava API-ból.")

    if _data_source == "garmin":
        st.info("Ez az elemzés Strava csatlakozás esetén aktív. Csatlakozz Stravához a bal oldali sávban.")
    else:
        _s = view.copy()
        _has_hr   = "hr_num" in _s.columns and _s["hr_num"].notna().sum() >= 3
        _has_pace = "pace_sec_km" in _s.columns and _s["pace_sec_km"].notna().sum() >= 3
        _has_cad  = "cad_num" in _s.columns and _s["cad_num"].notna().sum() >= 3
        _has_pwr  = "power_avg_w" in _s.columns and _s["power_avg_w"].notna().sum() >= 3
        _has_asc  = "asc_m" in _s.columns and _s["asc_m"].notna().sum() >= 3
        _has_suf  = "suffer_score" in _s.columns and _s["suffer_score"].notna().sum() >= 3
        _has_dist = "dist_km" in _s.columns and _s["dist_km"].notna().sum() >= 3

        # ── 1. KPI sor ──────────────────────────────────────────
        st.markdown("### 📊 Összesítők")
        _k = st.columns(5)
        _k[0].metric("Futások", f"{len(_s)}")
        _k[1].metric("Össz távolság",
                     f"{_s['dist_km'].sum():.0f} km" if _has_dist else "—")
        _k[2].metric("Össz idő",
                     f"{_s['dur_sec'].sum()/3600:.1f} h" if "dur_sec" in _s.columns else "—")
        _k[3].metric("Átlag HR",
                     f"{_s['hr_num'].mean():.0f} bpm" if _has_hr else "—")
        _k[4].metric("Össz emelkedés",
                     f"{_s['asc_m'].sum():.0f} m" if _has_asc else "—")

        st.divider()

        # ── 2. Tempó fejlődés ──────────────────────────────────
        if _has_pace and _has_dist:
            st.markdown("### 📈 Tempó fejlődés időben")
            _sp = _s[_s["pace_sec_km"].notna() & (_s["pace_sec_km"] > 0)].copy()
            _sp["tempó_label"] = _sp["pace_sec_km"].apply(sec_to_pace_str)
            _sp["pace_inv"] = 1000 / _sp["pace_sec_km"]

            _col1, _col2 = st.columns([3, 1])
            with _col1:
                _hover = {"tempó_label": True, "dist_km": ":.1f", "pace_inv": False}
                if _has_hr and "hr_num" in _sp.columns:
                    _hover["hr_num"] = True
                fig_pace = px.scatter(
                    _sp, x="Dátum", y="pace_inv",
                    size="dist_km", size_max=18,
                    color="hr_num" if _has_hr else None,
                    color_continuous_scale="RdYlGn_r",
                    hover_data=_hover,
                    labels={"pace_inv": "Sebesség (km/h)", "dist_km": "Táv (km)", "hr_num": "HR"},
                    title="Sebesség időben  (méretarány = távolság, szín = HR)",
                )
                if len(_sp) >= 8:
                    _sp2 = _sp.sort_values("Dátum").copy()
                    _sp2["roll"] = _sp2["pace_inv"].rolling(8, min_periods=4).mean()
                    for tr in px.line(_sp2, x="Dátum", y="roll",
                                      color_discrete_sequence=["#1a73e8"]).data:
                        tr.name = "8 futós átlag"; tr.showlegend = True
                        fig_pace.add_trace(tr)
                fig_pace.update_yaxes(tickformat=".2f")
                st.plotly_chart(fig_pace, use_container_width=True)
            with _col2:
                st.markdown("**Tempó statisztika**")
                for lbl, val in [
                    ("Leggyorsabb", sec_to_pace_str(_sp["pace_sec_km"].min())),
                    ("Leglassabb",  sec_to_pace_str(_sp["pace_sec_km"].max())),
                    ("Medián",      sec_to_pace_str(float(_sp["pace_sec_km"].median()))),
                    ("Utolsó 5 átl.", sec_to_pace_str(float(_sp.tail(5)["pace_sec_km"].mean()))),
                ]:
                    st.metric(lbl, val + " /km")

            st.divider()

        # ── 3. HR–Tempó – aerob hatékonyság ────────────────────
        if _has_hr and _has_pace:
            st.markdown("### ❤️ HR vs Tempó – aerob hatékonyság")
            st.caption("Minél lejjebb-jobbra egy pont, annál jobb: gyorsabb tempó alacsonyabb HR-rel.")
            _ht = _s[_s["hr_num"].notna() & _s["pace_sec_km"].notna() &
                     (_s["pace_sec_km"] > 0)].copy()
            _ht["pace_inv"] = 1000 / _ht["pace_sec_km"]
            hr_med = float(_ht["hr_num"].median())
            _ht["hr_zone"] = pd.cut(
                _ht["hr_num"],
                bins=[0, hr_med*0.80, hr_med*0.90, hr_med*1.0, hr_med*1.08, 999],
                labels=["Z1 Nagyon könnyű", "Z2 Könnyű", "Z3 Aerob", "Z4 Küszöb", "Z5 Intenzív"],
            )
            fig_ht = px.scatter(
                _ht, x="hr_num", y="pace_inv",
                color="hr_zone",
                size="dist_km" if _has_dist else None, size_max=16,
                hover_data={"Dátum": "|%Y-%m-%d", "dist_km": ":.1f"},
                labels={"hr_num": "Átlag HR (bpm)", "pace_inv": "Sebesség (km/h)", "hr_zone": "Zóna"},
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="HR vs Sebesség (zóna szerint színezve)",
            )
            _x_ht = _ht["hr_num"].to_numpy(dtype=float)
            _y_ht = _ht["pace_inv"].to_numpy(dtype=float)
            _mk = np.isfinite(_x_ht) & np.isfinite(_y_ht)
            if _mk.sum() >= 5:
                _cf = np.polyfit(_x_ht[_mk], _y_ht[_mk], 1)
                _xl = np.linspace(_x_ht[_mk].min(), _x_ht[_mk].max(), 50)
                fig_ht.add_scatter(x=_xl, y=np.polyval(_cf, _xl), mode="lines",
                                   name="Trend", line=dict(color="gray", dash="dash", width=1.5))
            st.plotly_chart(fig_ht, use_container_width=True)

            _ht["aei"] = _ht["pace_inv"] / _ht["hr_num"] * 100
            _n3 = max(1, len(_ht)//3)
            _aei_first = float(_ht.head(_n3)["aei"].mean())
            _aei_last  = float(_ht.tail(_n3)["aei"].mean())
            _aei_delta = _aei_last - _aei_first
            _ac1, _ac2, _ac3 = st.columns(3)
            _ac1.metric("AEI – első harmad",  f"{_aei_first:.2f}")
            _ac2.metric("AEI – utolsó harmad", f"{_aei_last:.2f}", delta=f"{_aei_delta:+.2f}")
            _ac3.metric("Trend",
                        "📈 Javuló" if _aei_delta > 0.05 else
                        "📉 Romló"  if _aei_delta < -0.05 else "➡️ Stabil")

            st.divider()

        # ── 4. Heti volumen & ramp rate ────────────────────────
        if _has_dist:
            st.markdown("### 📅 Heti volumen & ramp rate")
            _sw = _s.copy()
            _sw["hét"] = _sw["Dátum"].dt.to_period("W").dt.start_time
            _weekly = _sw.groupby("hét").agg(
                km=("dist_km", "sum"),
                futások=("dist_km", "count"),
                hr_avg=("hr_num", "mean"),
                asc=("asc_m", "sum"),
            ).reset_index()
            _weekly["prev_km"] = _weekly["km"].shift(1)
            _weekly["ramp"] = ((_weekly["km"] - _weekly["prev_km"])
                               / _weekly["prev_km"].replace(0, np.nan) * 100)

            _wc1, _wc2 = st.columns(2)
            with _wc1:
                fig_wk = px.bar(_weekly, x="hét", y="km",
                                color="hr_avg" if _has_hr else None,
                                color_continuous_scale="RdYlGn_r",
                                labels={"km": "Heti km", "hét": "", "hr_avg": "Átlag HR"},
                                title="Heti futott kilométerek")
                st.plotly_chart(fig_wk, use_container_width=True)
            with _wc2:
                _ramp_df = _weekly.dropna(subset=["ramp"]).copy()
                if len(_ramp_df) >= 3:
                    _ramp_df["szín"] = _ramp_df["ramp"].apply(
                        lambda r: "piros" if r > CFG["ramp_red"]
                        else "sárga" if r > CFG["ramp_warn"] else "zöld"
                    )
                    fig_ramp = px.bar(_ramp_df, x="hét", y="ramp", color="szín",
                                      color_discrete_map={"piros": "#e74c3c",
                                                          "sárga": "#f39c12", "zöld": "#2ecc71"},
                                      title="Heti ramp rate (%)",
                                      labels={"ramp": "Ramp (%)", "hét": ""})
                    fig_ramp.add_hline(y=CFG["ramp_warn"], line_dash="dot",
                                       line_color="orange", annotation_text="Figyelj (8%)")
                    fig_ramp.add_hline(y=CFG["ramp_red"], line_dash="dot",
                                       line_color="red", annotation_text="Veszélyes (12%)")
                    st.plotly_chart(fig_ramp, use_container_width=True)
                else:
                    st.info("Ramp rate-hez legalább 4 hét adat kell.")

            st.divider()

        # ── 5. Kadencia ────────────────────────────────────────
        if _has_cad:
            st.markdown("### 🦵 Kadencia elemzés")
            _sc = _s[_s["cad_num"].notna() & (_s["cad_num"] > 100)].copy()
            _cc1, _cc2 = st.columns(2)
            with _cc1:
                fig_cad = px.scatter(
                    _sc, x="Dátum", y="cad_num",
                    color="hr_num" if _has_hr else None,
                    color_continuous_scale="RdYlGn_r",
                    title="Kadencia időben",
                    labels={"cad_num": "Kadencia (spm)", "hr_num": "HR"},
                )
                fig_cad.add_hline(y=170, line_dash="dot", line_color="orange",
                                  annotation_text="170 spm ajánlott alsó")
                fig_cad.add_hline(y=180, line_dash="dot", line_color="green",
                                  annotation_text="180 spm optimum")
                if len(_sc) >= 8:
                    _sc2 = _sc.sort_values("Dátum").copy()
                    _sc2["roll"] = _sc2["cad_num"].rolling(8, min_periods=4).mean()
                    for tr in px.line(_sc2, x="Dátum", y="roll",
                                      color_discrete_sequence=["#1a73e8"]).data:
                        tr.name = "8 futós átlag"; fig_cad.add_trace(tr)
                st.plotly_chart(fig_cad, use_container_width=True)
            with _cc2:
                fig_cad_hist = px.histogram(_sc, x="cad_num", nbins=20,
                                            title="Kadencia eloszlás",
                                            labels={"cad_num": "Kadencia (spm)"},
                                            color_discrete_sequence=["#3498db"])
                fig_cad_hist.add_vline(x=180, line_dash="dash", line_color="green",
                                       annotation_text="180 spm")
                st.plotly_chart(fig_cad_hist, use_container_width=True)
            _cad_ok = (_sc["cad_num"] >= 170).mean() * 100
            _cdc1, _cdc2, _cdc3 = st.columns(3)
            _cdc1.metric("Átlag kadencia",   f"{_sc['cad_num'].mean():.0f} spm")
            _cdc2.metric("Medián kadencia",  f"{_sc['cad_num'].median():.0f} spm")
            _cdc3.metric("Futások 170+ spm", f"{_cad_ok:.0f}%")
            st.divider()

        # ── 6. Emelkedés ───────────────────────────────────────
        if _has_asc and _has_dist:
            st.markdown("### ⛰️ Emelkedés és terep")
            _sa = _s[_s["dist_km"].notna() & (_s["dist_km"] > 0)].copy()
            _sa["emelkedés/km"] = _sa["asc_m"] / _sa["dist_km"]
            _asc1, _asc2 = st.columns(2)
            with _asc1:
                fig_asc = px.scatter(
                    _sa, x="dist_km", y="asc_m",
                    color="pace_sec_km" if _has_pace else None,
                    color_continuous_scale="RdYlGn",
                    size="dur_sec" if "dur_sec" in _sa.columns else None, size_max=18,
                    hover_data={"Dátum": "|%Y-%m-%d", "emelkedés/km": ":.1f"},
                    title="Emelkedés vs távolság",
                    labels={"dist_km": "Távolság (km)", "asc_m": "Emelkedés (m)"},
                )
                st.plotly_chart(fig_asc, use_container_width=True)
            with _asc2:
                if "slope_bucket" in _sa.columns:
                    _terep = _sa["slope_bucket"].apply(slope_bucket_hu).value_counts().reset_index()
                    _terep.columns = ["Terep", "Futások"]
                    fig_terep = px.pie(_terep, values="Futások", names="Terep",
                                       title="Terep megoszlás",
                                       color_discrete_sequence=px.colors.qualitative.Safe)
                    st.plotly_chart(fig_terep, use_container_width=True)
            st.divider()

        # ── 7. Suffer Score ────────────────────────────────────
        if _has_suf:
            st.markdown("### 😤 Suffer Score – edzésterhelés")
            _ss = _s[_s["suffer_score"].notna() & (_s["suffer_score"] > 0)].copy()
            _sfc1, _sfc2 = st.columns(2)
            with _sfc1:
                fig_suf = px.bar(_ss.sort_values("Dátum"), x="Dátum", y="suffer_score",
                                 color="suffer_score", color_continuous_scale="RdYlGn_r",
                                 title="Suffer Score futásonként",
                                 labels={"suffer_score": "Suffer Score"})
                if len(_ss) >= 6:
                    _ss2 = _ss.sort_values("Dátum").copy()
                    _ss2["roll"] = _ss2["suffer_score"].rolling(6, min_periods=3).mean()
                    for tr in px.line(_ss2, x="Dátum", y="roll",
                                      color_discrete_sequence=["#2c3e50"]).data:
                        tr.name = "6 futós átlag"; fig_suf.add_trace(tr)
                st.plotly_chart(fig_suf, use_container_width=True)
            with _sfc2:
                if _has_hr:
                    fig_suf_hr = px.scatter(
                        _ss, x="hr_num", y="suffer_score",
                        size="dist_km" if _has_dist else None, size_max=16,
                        color="pace_sec_km" if _has_pace else None,
                        color_continuous_scale="RdYlGn",
                        title="HR vs Suffer Score",
                        labels={"hr_num": "Átlag HR (bpm)", "suffer_score": "Suffer Score"},
                    )
                    _x_sf = _ss["hr_num"].to_numpy(dtype=float)
                    _y_sf = _ss["suffer_score"].to_numpy(dtype=float)
                    _mf = np.isfinite(_x_sf) & np.isfinite(_y_sf)
                    if _mf.sum() >= 5:
                        _cf2 = np.polyfit(_x_sf[_mf], _y_sf[_mf], 1)
                        _xl2 = np.linspace(_x_sf[_mf].min(), _x_sf[_mf].max(), 50)
                        fig_suf_hr.add_scatter(x=_xl2, y=np.polyval(_cf2, _xl2),
                                               mode="lines", name="Trend",
                                               line=dict(color="gray", dash="dash"))
                    st.plotly_chart(fig_suf_hr, use_container_width=True)
                else:
                    fig_suf_hist = px.histogram(_ss, x="suffer_score", nbins=15,
                                                title="Suffer Score eloszlás",
                                                color_discrete_sequence=["#e74c3c"])
                    st.plotly_chart(fig_suf_hist, use_container_width=True)
            _sk1, _sk2, _sk3, _sk4 = st.columns(4)
            _sk1.metric("Átlag",  f"{_ss['suffer_score'].mean():.0f}")
            _sk2.metric("Max",    f"{_ss['suffer_score'].max():.0f}")
            _sk3.metric("Össz",   f"{_ss['suffer_score'].sum():.0f}")
            _sk4.metric("Intenzív (>50)", f"{(_ss['suffer_score'] > 50).sum()}")
            st.divider()

        # ── 8. Power (Stryd) ───────────────────────────────────
        if _has_pwr:
            st.markdown("### ⚡ Power – futás-gazdaságosság (Stryd)")
            _sp_p = _s[_s["power_avg_w"].notna() & (_s["power_avg_w"] > 0)].copy()
            _pw1, _pw2 = st.columns(2)
            with _pw1:
                fig_pwr = px.scatter(
                    _sp_p, x="Dátum", y="power_avg_w",
                    size="dist_km" if _has_dist else None, size_max=18,
                    color="hr_num" if _has_hr else None,
                    color_continuous_scale="RdYlGn_r",
                    title="Átlag power időben",
                    labels={"power_avg_w": "Power (W)", "hr_num": "HR"},
                )
                if len(_sp_p) >= 8:
                    _sp_p2 = _sp_p.sort_values("Dátum").copy()
                    _sp_p2["roll"] = _sp_p2["power_avg_w"].rolling(8, min_periods=4).mean()
                    for tr in px.line(_sp_p2, x="Dátum", y="roll",
                                      color_discrete_sequence=["#8e44ad"]).data:
                        tr.name = "8 futós átlag"; fig_pwr.add_trace(tr)
                st.plotly_chart(fig_pwr, use_container_width=True)
            with _pw2:
                if _has_hr:
                    _sp_p["pw_hr"] = _sp_p["power_avg_w"] / _sp_p["hr_num"]
                    fig_pwhr = px.scatter(
                        _sp_p, x="Dátum", y="pw_hr",
                        title="Power/HR arány (futás-gazdaságosság proxy)",
                        labels={"pw_hr": "W/bpm"},
                        color_discrete_sequence=["#27ae60"],
                    )
                    if len(_sp_p) >= 8:
                        _sp_p["roll_r"] = _sp_p["pw_hr"].rolling(8, min_periods=4).mean()
                        for tr in px.line(_sp_p, x="Dátum", y="roll_r",
                                          color_discrete_sequence=["#2c3e50"]).data:
                            tr.name = "Trend"; fig_pwhr.add_trace(tr)
                    st.plotly_chart(fig_pwhr, use_container_width=True)

        # ── 9. Hiányzó adatok figyelmeztetése ──────────────────
        _miss = [n for n, h in [("Pulzus (HR)", _has_hr), ("Kadencia", _has_cad),
                                 ("Power (Stryd)", _has_pwr), ("Suffer Score", _has_suf)]
                 if not h]
        if _miss:
            st.info(
                f"⚠️ Hiányzó adatok ebben az időszakban: **{', '.join(_miss)}**. "
                f"Pulzusmérő és Stryd eszközök bővítik az elemzést."
            )


# =========================================================
# TAB: ADATOK
# =========================================================
with tab_data:
    st.subheader("📄 Adatok (szűrve)")
    st.caption("A szűrt futások teljes adattáblája. Az elemzésekhez elég az első 4 tab.")
    st.dataframe(view, use_container_width=True, hide_index=True, height=520)
