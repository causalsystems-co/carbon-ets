CS/RES/05 · 7 July 2026 · 4 min read

# Factories to carbon.

A demand-side model of EU allowance prices, and the specific regime it fails in.

EU carbon looks like a policy market, and mostly it is. But the demand side is very physical: regulated installations emit in proportion to what they run, and what they run scales with industrial production, thermal-power demand, and the marginal fuel choice. We wanted to know how much of the EUA price the physical chain actually explains — and where the mechanism breaks down.

Industrial production → Compliance demand → EUA auction clearing · monthly · 45 d lag · daily print

Fourteen-and-a-half years of daily EEX primary-auction clearing prices (2012–2026, n ≈ 3,600) against Eurostat industrial production for the euro area, Stoxx 50 momentum as a high-frequency sentiment proxy, and Open-Meteo Frankfurt heating-degree-days. Free public sources throughout. The headline is deliberately narrow: a two-feature demand model (z-scored IP YoY plus z-scored Stoxx momentum) explains only 5% of monthly EUA return variance on the full sample — but reaches 63% in the specific 2021 window when demand fundamentals dominated. The chain doesn't run continuously; it activates in specific regimes.

Plot 01 · Rolling monthly R² of the two-feature demand model, 24-month window, 2017–2025.

The mechanism is straightforward. When European factories run harder they emit more, compliance buyers must acquire more allowances against annual surrender, and the auction absorbs the incremental demand at a higher clearing price. Because corporate procurement is budgeted rather than algorithmic, the demand signal shows up in the tape with a lag — five to twenty days on daily returns, a full month on monthly aggregates. IP is a monthly release with a 45-day lag; Stoxx moves daily. They tell the same story at different frequencies.

R² monthly

0.05

full sample · 175 obs

R² peak

0.63

late 2021 · 24-mo window

Sign-correct years

12 / 12

2014–2025 · every year positive

The 2018-2019 window shows the chain at its weakest. Between January 2018 and July 2019, EUA rose from €8 to €25 while IP YoY was essentially flat. That was the Market Stability Reserve reform — a supply-side news event that the demand model has no visibility into. Rolling R² over that period drops to near zero. A similar collapse happens on a smaller scale during the 2014-2017 oversupply hangover, where a mild European industrial recovery couldn't move a price pinned near €5 by structural surplus. The chain is inactive again from 2023 onward as policy dynamics — MSR invalidation, CBAM implementation, ETS2 preparation — crowd out fundamentals.

Plot 02 · EUA price vs euro-area IP YoY, 2012–2026, twin axis. The chain co-moves through most of the sample and separates sharply during 2018.

The failure is not a bug. It's the boundary condition. EU ETS is a policy-managed market in which supply is administratively fixed and demand is physically determined. The demand side is what our chain models. The supply side — MSR intake rates, free-allowance revisions, CBAM, ETS2, REPowerEU sales — arrives as discrete announcements from the Commission. Any complete model has to include a policy channel. The interesting thing is that the boundary is diagnosable: when TNAC crosses the MSR threshold, or when a Commission consultation opens, the regime is about to shift, and the demand-only model should be down-weighted before the fact rather than after.

We tested that directly. Reading the TNAC series from all ten Commission Communications published under the MSR mechanism, the surplus falls from 1.69 billion allowances at end-2016 to 1.02 billion at end-2025 — a 40% structural tightening in nine years. Crossing below the 1.096B threshold in 2025 activated the partial-intake regime for the first time in the mechanism's history. Adding TNAC as a third feature to the demand model closed some of the 2020+ variance but did not recover the 2018 anomaly. The reason is causal: TNAC as a level captures where the surplus stands, but the 2018 rally was driven by expectations of *future* MSR intake, legislated but not yet mechanically applied. That distinction is where the next research needs to focus — anticipation of policy changes, not their measured realisation.

Plot 03 · TNAC series with MSR regime thresholds annotated. 2025 crosses the partial-intake band for the first time.

Two smaller findings worth flagging. Monthly cross-correlation between IP YoY and EUA returns peaks at lag 10–20 days, small in absolute terms (~0.05) but stable across the sample. And splitting daily EUA returns by IP YoY sign gives the expected asymmetry: expansion months annualise +18% at 32% vol, contraction months −6% at 40% vol. The mechanism is real; the risk premium sits in the expansion leg.

The framework is easy to extend. The demand-side chain modelled here is the first of a family — sister models on European power, spark spreads, TTF gas, and (in preparation) ETS2 share the same physical inputs and the same methodology. Together they describe most of the short-run price formation across the European energy complex. Each chain plugs into the same pipeline, so a policy change modelled in one flows through to the others.

Everything in this piece is reproducible. The pipeline that produced every figure — EEX auction fetcher, Eurostat IP fetcher, verified TNAC series with primary-source citations per Commission Communication, feature engineering, and the backtest itself — is released as an open-source Python toolkit at [github.com/causalsystems-co/carbon-ets](https://github.com/causalsystems-co/carbon-ets). `pip install carbon-ets` gives you the reference data and the models we describe here. The single-line CLI `carbon-ets-tnac` prints the verified TNAC series with its current MSR regime, sourced from Commission PDFs, in less than a second.

> The chain is real. It just has a boundary — and that boundary is diagnosable in advance.

---

Causal *Systems* · CS/RES/05 · Sources: EEX primary auctions (2012–2026), Eurostat SDMX (industrial production), Yahoo Finance (equities), Open-Meteo (weather), European Commission MSR Communications C(2017) 3228 through OJ C_202602957.
