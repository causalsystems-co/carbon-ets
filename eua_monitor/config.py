"""Configuration for the EUA regime monitor.

Everything tunable lives here: data sources, model window, regime cutoffs,
MSR thresholds, and the HDD city basket. Values follow CS/RES/05
("Factories to Carbon", causalsystems.co/research/factories-to-carbon).
"""

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
SITE_DIR = PKG_DIR.parent / "site"

# ---------------------------------------------------------------- data sources
# Primary EUA series: EEX primary-auction clearing prices (as in CS/RES/05).
# data/eua_history.csv holds the imported 2012+ archive (see eex_import.py);
# the current year tops up live from the public EEX report. The WisdomTree
# Carbon ETC (CO2.L) is only a recency fallback when the EEX feed is stale.
EUA_TICKER = "CO2.L"
STOXX_TICKER = "^STOXX50E"
EUA_HISTORY_CSV = DATA_DIR / "eua_history.csv"
EEX_CURRENT_URL_TMPL = (
    "https://public.eex-group.com/eex/eua-auction-report/"
    "emission-spot-primary-market-auction-report-{year}-data.xlsx"
)
EEX_STALE_DAYS = 10  # append proxy tail if the auction feed is older than this

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Eurostat: euro-area industrial production, monthly, seasonally and
# calendar adjusted, index 2021=100, industry ex construction (B-D).
EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "sts_inpr_m?format=JSON&geo=EA20&nace_r2=B-D&s_adj=SCA&unit=I21"
    "&sinceTimePeriod=2011-01"
)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Population-weighted HDD basket (base 18 C), covering the big EU demand zones.
HDD_CITIES = [
    ("Frankfurt", 50.11, 8.68, 0.25),
    ("Paris", 48.86, 2.35, 0.20),
    ("Milan", 45.46, 9.19, 0.20),
    ("Madrid", 40.42, -3.70, 0.15),
    ("Warsaw", 52.23, 21.01, 0.10),
    ("Amsterdam", 52.37, 4.90, 0.10),
]
HDD_START = "2012-01-01"
HDD_BASE_C = 18.0

# ---------------------------------------------------------------------- model
ROLLING_WINDOW_MONTHS = 24
STOXX_MOMENTUM_MONTHS = 3

# Regime cutoffs on the rolling-window R^2 of the demand model.
R2_DEMAND_DRIVEN = 0.35   # >= : demand fundamentals dominate
R2_POLICY_DRIVEN = 0.15   # <= : policy/positioning dominate
# in between: transitional

# ----------------------------------------------------------------- MSR / TNAC
# Decision (EU) 2015/1814 as amended by Decision (EU) 2023/852:
#   TNAC >= 1,096M            -> intake = 24% of TNAC
#   833M < TNAC < 1,096M      -> intake = TNAC - 833M (partial intake)
#   400M <= TNAC <= 833M      -> no action
#   TNAC < 400M               -> 100M released from the reserve
MSR_UPPER_FULL = 1_096_000_000
MSR_UPPER = 833_000_000
MSR_LOWER = 400_000_000
MSR_INTAKE_RATE = 0.24
MSR_RELEASE = 100_000_000

TNAC_CSV = DATA_DIR / "tnac.csv"

# ------------------------------------------------------------------- auctions
# Indicative EEX weekday pattern for EUA primary auctions; the published EEX
# auction calendar governs. Exact rows can be supplied via
# data/auction_overrides.csv (columns: date,market,note).
AUCTION_WEEKDAYS = {
    0: "EU ETS — EU27 common auction",
    1: "EU ETS — EU27 common auction",
    2: "EU ETS — Germany (DE)",
    3: "EU ETS — EU27 common auction",
    4: "EU ETS — Poland (PL, biweekly)",
}
AUCTION_DAYS_AHEAD = 14
AUCTION_OVERRIDES_CSV = DATA_DIR / "auction_overrides.csv"

# -------------------------------------------------------------------- outputs
STATE_JSON = SITE_DIR / "state.json"
DASHBOARD_HTML = SITE_DIR / "index.html"
DIGEST_MD = SITE_DIR / "digest.md"

RESEARCH_URL = "https://causalsystems.co/research/factories-to-carbon"
TOOLKIT_URL = "https://github.com/causalsystems/carbon-ets"
CONTACT_MAILTO = (
    "mailto:research@causalsystems.co?subject=EUA%20regime%20digest%20"
    "subscription&body=Please%20add%20me%20to%20the%20EUA%20regime%20digest."
)
