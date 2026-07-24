#!/usr/bin/env python3
"""Retheme the built EUA monitor page to the causalsystems.co design.

Run after `python -m eua_monitor build`, before deploying:
    python restyle_cs.py site/index.html
Idempotent: applying twice changes nothing.
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "site/index.html"
s = open(path, encoding="utf-8").read()

if "CS-THEME" in s:
    print("already themed, nothing to do")
    sys.exit(0)

# ---- charset + robots + fonts + favicon (idempotent marker in comment) ----
s = s.replace(
    "<!doctype html>\n<title>",
    "<!doctype html>\n<!-- CS-THEME -->\n<meta charset=\"utf-8\">\n<title>",
    1,
)
s = s.replace(
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    '<meta name="robots" content="noindex,nofollow">\n'
    '<link rel="icon" type="image/svg+xml" href="../../assets/favicon.svg">\n'
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800;900'
    '&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">',
    1,
)

# ---- light palette -> CS ----
s = s.replace(
    """--page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --hairline: rgba(11,11,11,0.10);
  --series: #2a78d6; --series-wash: rgba(42,120,214,0.10);
  --good: #0ca30c; --warn: #fab219; --serious: #ec835a;
  --good-wash: rgba(12,163,12,0.11); --warn-wash: rgba(250,178,25,0.13);
  --serious-wash: rgba(236,131,90,0.13);""",
    """--page: #F4F1EA; --surface: #EAE3D5;
  --ink: #141414; --ink-2: #1F2937; --muted: #6B7280;
  --grid: #d9d3c3; --baseline: #b8b2a2; --hairline: rgba(20,20,20,0.15);
  --series: #2B53FF; --series-wash: rgba(43,83,255,0.10);
  --good: #008300; --warn: #B57A00; --serious: #C33A12;
  --good-wash: rgba(0,131,0,0.11); --warn-wash: rgba(181,122,0,0.13);
  --serious-wash: rgba(195,58,18,0.12);""",
)

# ---- both dark blocks -> CS ink ----
dark_new = """color-scheme: dark;
    --page: #141414; --surface: #1f1e1b;
    --ink: #F5F1E8; --ink-2: #EAE3D5; --muted: #8A8275;
    --grid: #2e2c28; --baseline: #3a3833; --hairline: rgba(255,255,255,0.12);
    --series: #7C9AFF; --series-wash: rgba(124,154,255,0.14);
    --good: #4CAF50; --warn: #E0A93E; --serious: #E8734D;
    --good-wash: rgba(76,175,80,0.15); --warn-wash: rgba(224,169,62,0.14);
    --serious-wash: rgba(232,115,77,0.15);"""
s = re.sub(
    r"color-scheme: dark;\s*\n\s*--page: #0d0d0d;.*?--serious-wash: rgba\(236,131,90,0\.14\);",
    dark_new, s, flags=re.S,
)

# ---- fonts ----
s = s.replace(
    'font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;',
    'font: 15px/1.55 "Archivo", system-ui, -apple-system, "Segoe UI", sans-serif;',
)
s = s.replace(
    "font: 600 13px/1 system-ui, sans-serif",
    'font: 600 13px/1 "Archivo", system-ui, sans-serif',
)
s = s.replace(
    "ui-monospace, SFMono-Regular, Menlo, monospace",
    '"Space Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
)

# ---- brand header with CS mark ----
s = s.replace(
    '<div class="wordmark">Causal Systems <span>/ EUA Regime Monitor</span></div>',
    '<div class="wordmark" style="display:flex;align-items:center;gap:9px">'
    '<svg viewBox="-2 18 116 86" style="height:24px;width:auto;overflow:visible" role="img" aria-label="Causal Systems">'
    '<text x="2" y="90" font-family="Archivo, sans-serif" font-weight="900" font-size="92" letter-spacing="-4" fill="currentColor">C</text>'
    '<text x="48" y="96" font-family="Archivo, sans-serif" font-weight="900" font-size="92" letter-spacing="-4" fill="#FF4A1C">S</text></svg>'
    '<span style="font-weight:800">Causal Systems</span> <span>/ EUA Regime Monitor</span></div>',
)

# ---- house style: no em dashes ----
s = s.replace("&mdash;", "·").replace("—", " · ").replace(" ·  · ", " · ")

open(path, "w", encoding="utf-8").write(s)
print("themed:", path)
