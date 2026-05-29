# Carbon_ETS — EU ETS (EUA) Causal Chain

Sister project to `EU_Power_Model/`, `Spark_Spread/`, `TTF_Gas/`.

EU Allowances (EUAs) are the most policy-driven major commodity in Europe.
Price is set by the intersection of *capped supply* and *demand from regulated
emitters*. Demand moves with industrial production, electricity generation
mix, and weather. Supply moves with auction calendars and Market Stability
Reserve (MSR) rules. Policy news (CBAM, ETS2, free-allowance changes)
re-prices the curve in seconds.

This makes EUA an excellent causal-chain candidate: the upstream drivers
(production, electricity, weather, fuel switching, policy) all publish
*before or independently* of the price, on schedules that don't depend on
the market.

## The mechanism in one sentence

When industrial output and thermal-power generation rise, compliance buyers
must purchase more EUAs to cover emissions; when output falls (recession,
mild weather, gas-to-renewables switching), compliance demand collapses
and the EUA curve sells off — and most of the upstream signal publishes
days to weeks before the EUA price reacts.

## The chain

```
Macro / IP                Electricity demand        Fuel switching
(Eurostat IP, PMI)        (ENTSO-E load, weather)   (TTF gas / API2 coal / EUA)
        │                          │                          │
        └──────────┬───────────────┴──────────┬───────────────┘
                   ▼                          ▼
            Verified emissions          Marginal generator
            (EUTL, annual + monthly     (gas vs coal → tCO2/MWh
             power-sector proxies)       intensity changes)
                   │                          │
                   └─────────────┬────────────┘
                                 ▼
                        EUA compliance demand
                                 │
                 Auction calendar (EEX, weekly)
                 MSR intake / TNAC (annual May 15)
                                 │
                                 ▼
                        EUA front-month price
                                 │
                  Policy shocks (CBAM, ETS2, free
                  allowances, REPowerEU sales)
                                 │
                                 ▼
                   Tradable: ICE EUA futures,
                            KRBN/GRN ETFs (proxy),
                            calendar spreads,
                            options on EUA
```

## Why this is causally tradable

1. **Demand-side leading indicators publish first.** ENTSO-E publishes
   day-ahead load 12 hours before the power market clears. Eurostat IP
   releases ~45 days after month-end but with stable seasonality.
   Weather forecasts run 10 days out. PMIs publish first business day
   of the month. EUA only reflects all of this with a lag, because
   compliance purchasing is *budgeted, not algorithmic*.

2. **Supply is calendar-locked.** EEX auctions every Mon-Thu morning,
   ~14:00 CET clearing. MSR intake rate updates published May 15
   annually. Free-allocation changes telegraphed in advance via
   Commission press releases. Supply shocks come from *policy*, not
   from auctions clearing high/low — and policy is observable.

3. **The recession leg is the cleanest.** In every prior contraction
   (2008-09, 2020 COVID, 2022 gas-crisis demand destruction) EUAs
   sold off 30-60% with a 1-3 month lag to IP. The mechanism is
   physical: factories that don't run don't emit.

## Folder layout

```
Carbon_ETS/
├── README.md                  ← this file
├── MAURIZIO_HANDOFF.md        ← onboarding for the new contributor
├── requirements.txt
├── data/                      ← parquet outputs, gitignored
├── plots/                     ← png outputs
├── notebooks/                 ← exploration
└── scripts/
    ├── 01_fetch_prices.py     ← EUA proxy + fuels + macro tickers
    ├── 02_fetch_emissions.py  ← IP, ENTSO-E load, weather, generation mix
    ├── 03_build_dataset.py    ← merge all sources to daily panel
    ├── 04_analyze_chain.py    ← lead-lag, regression, IRF, regime split
    └── 05_backtest.py         ← signal → position → equity curve
```

## Run order

```bash
cd Carbon_ETS
pip install -r requirements.txt
python scripts/01_fetch_prices.py     --start 2018-01-01
python scripts/02_fetch_emissions.py  --start 2018-01-01
python scripts/03_build_dataset.py
python scripts/04_analyze_chain.py
python scripts/05_backtest.py
```

Each script is independent and writes to `data/`. Re-running 05 doesn't
require re-fetching upstream data.

## What's stubbed vs. what's real

The `01_fetch_prices.py` and `02_fetch_emissions.py` scripts use **only
free public data** (yfinance, ENTSO-E public API, Eurostat, Open-Meteo).
Where the *real* feed is paid (ICE EUA settlement, S&P Global emissions),
the scripts use the best free proxy and clearly flag the substitution.

The `04_analyze_chain.py` and `05_backtest.py` scripts are intentionally
**minimal but complete** — they run end-to-end and produce sensible
outputs, but every model choice (lag windows, signal construction,
position sizing) is a `TODO` for Maurizio to extend.

## Things for Maurizio to try

Ranked by expected payoff.

1. Replace KRBN proxy with real ICE EUA front-month (need a Refinitiv /
   Bloomberg / ICE API key, or scrape EEX auction results daily — free).
2. Add the EEX primary-auction clearing price as a separate series and
   test whether auction-day premium/discount to secondary market
   predicts next-day direction.
3. Build the *compliance calendar* feature: April 30 surrender deadline,
   May 15 MSR announcement, December position-squaring. Each is a
   known seasonal that the current code does not exploit.
4. Add a CBAM / ETS2 / policy-news classifier. The headline text is
   on EU Commission press release feeds; sentiment + topic
   classification on those is a strong feature.
5. Cross-asset: EUA vs UKA spread (post-Brexit divergence), EUA vs
   CCA (California), EUA vs RGGI — all tradable in ETF form.

## Sister chains in this repo

| Folder | Driver | Tradable | Correlation w/ EUA expected |
|---|---|---|---|
| `EU_Power_Model/` | Weather → power price | DE/FR day-ahead | ~0.3 (shared weather) |
| `Spark_Spread/` | Fuel switching → power | Clean spark/dark | ~0.5 (shared carbon) |
| `TTF_Gas/` | Gas storage → TTF | TTF futures | ~0.4 (gas-to-EUA pass-through) |

EUA is the natural overlay across all three.
