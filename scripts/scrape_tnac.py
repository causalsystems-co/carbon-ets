"""
scrape_tnac.py — pull TNAC 2021-2025 from EUR-Lex HTML.

One-off helper. Prints the extracted TNAC values so you can paste them
into 02c_fetch_msr_tnac.py.
"""
import re, requests

URLS = {
    2021: "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:52022XC0513(01)",
    2022: "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:52023XC0515(01)",
    2023: "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:C_202403415",
    2024: "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:C_202503180",
    2025: "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:C_202602957",
}

# Numbers in EUR-Lex tables can be formatted "1 449 000 000" (spaces) or
# "1,449,000,000". Match either; require >= 800M and <= 2B (TNAC range).
NUM_RE = re.compile(r"([0-9](?:[\s, ][0-9]{3}){2,})")

for year, url in URLS.items():
    try:
        html = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"}).text
    except Exception as e:
        print(f"{year}: fetch failed — {e}")
        continue

    # find TNAC-magnitude numbers (order 10^9)
    candidates = []
    for m in NUM_RE.finditer(html):
        raw = m.group(1).replace(",", "").replace(" ", "").replace(" ", "")
        try:
            n = int(raw)
            if 800_000_000 <= n <= 2_500_000_000:
                candidates.append(n)
        except ValueError:
            pass

    if not candidates:
        print(f"{year}: no candidate numbers — inspect {url} manually")
        continue

    # TNAC is usually the last number in the concluding overview table,
    # and it's usually stated 2-3 times in the doc. Report all uniques.
    uniq = sorted(set(candidates))
    print(f"{year} candidates: {[f'{n:,}' for n in uniq]}")
    print(f"   most likely TNAC: {max(candidates):,}    (last occurrence)")
    print()
