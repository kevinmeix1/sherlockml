"""Render docs/demo/DEMO_WALKTHROUGH.md into a themed demo PDF.

Usage (from the repository root):

    python3 -m pip install markdown playwright
    python3 -m playwright install chromium
    python3 docs/demo/make_pdf.py

The output is written next to this script as ``SherlockML-Demo-Walkthrough.pdf``.
"""

# ruff: noqa: E501

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "DEMO_WALKTHROUGH.md"
OUTPUT = HERE / "SherlockML-Demo-Walkthrough.pdf"

_CSS = """
  @page { size: A4; margin: 0; }
  * { box-sizing: border-box; }
  :root {
    --ink: #07131b; --panel: #0d222a; --paper: #f5f1e8; --muted: #9caab0;
    --line: rgba(190, 215, 211, .18); --mint: #5de0b5; --coral: #ff7462;
    --gold: #f2c86f; --blue: #89c7ff;
  }
  html, body { margin: 0; padding: 0; background: var(--ink); }
  body { font-family: 'DM Sans', 'Segoe UI', Helvetica, Arial, sans-serif; color: #d5e0df; font-size: 10.5pt; line-height: 1.55; }
  .cover { height: 296.5mm; display: flex; flex-direction: column; justify-content: center; padding: 22mm 20mm;
           background: radial-gradient(circle at 15% 0%, #17394a 0, transparent 40%), radial-gradient(circle at 95% 15%, #213c36 0, transparent 34%), var(--ink);
           page-break-after: always; }
  .cover .eyebrow { color: var(--mint); font-family: 'DM Mono', 'Courier New', monospace; font-size: 9pt; letter-spacing: .16em; text-transform: uppercase; }
  .cover h1 { font-family: 'Playfair Display', Georgia, serif; color: var(--paper); font-size: 34pt; margin: .4rem 0 .6rem; letter-spacing: -.02em; }
  .cover .sub { color: #b7c8c7; font-size: 12.5pt; max-width: 130mm; }
  .cover .chips { margin-top: 10mm; }
  .chip { display: inline-block; margin: 0 2mm 2mm 0; padding: 1.4mm 3.2mm; border: 1px solid rgba(93,224,181,.4); border-radius: 99px;
          color: var(--mint); font-family: 'DM Mono', monospace; font-size: 7.5pt; letter-spacing: .08em; }
  .chip.blue { border-color: rgba(137,199,255,.45); color: var(--blue); }
  .chip.gold { border-color: rgba(242,200,111,.45); color: var(--gold); }
  .cover .date { margin-top: 16mm; color: var(--muted); font-family: 'DM Mono', monospace; font-size: 8.5pt; }
  main { padding: 14mm 16mm 16mm; }
  h1, h2, h3 { font-family: 'Playfair Display', Georgia, serif; color: var(--paper); letter-spacing: -.02em; }
  main > h1 { display: none; }              /* the cover replaces the markdown H1 */
  main > blockquote:first-of-type { display: none; }
  h2 { font-size: 17pt; margin: 0 0 4mm; padding-bottom: 2mm; border-bottom: 1px solid var(--line); page-break-before: always; padding-top: 6mm; }
  h2:first-of-type { page-break-before: avoid; padding-top: 0; }
  h3 { font-size: 12.5pt; color: var(--mint); margin: 6mm 0 2mm; }
  p { margin: 2.2mm 0; }
  a { color: var(--blue); text-decoration: none; }
  strong { color: var(--paper); }
  em { color: var(--gold); font-style: italic; }
  ul, ol { margin: 2mm 0 3mm; padding-left: 6mm; }
  li { margin: 1.4mm 0; }
  li::marker { color: var(--mint); }
  img { width: 100%; border: 1px solid var(--line); border-radius: 8px; margin: 3mm 0; page-break-inside: avoid; }
  code { font-family: 'DM Mono', 'Courier New', monospace; font-size: 8.8pt; color: var(--mint); background: rgba(93,224,181,.08);
         border: 1px solid rgba(93,224,181,.18); border-radius: 4px; padding: .2mm 1.2mm; }
  pre { background: #040d12; border: 1px solid rgba(93,224,181,.25); border-radius: 8px; padding: 3.5mm 4mm; overflow-x: hidden;
        page-break-inside: avoid; }
  pre code { background: none; border: none; padding: 0; color: #c5e8dc; font-size: 8.6pt; line-height: 1.6; }
  blockquote { margin: 3mm 0; padding: 2.5mm 4mm; border-left: 3px solid var(--gold); background: rgba(242,200,111,.06);
               border-radius: 0 6px 6px 0; color: #e3d9c4; }
  blockquote p { margin: 1mm 0; }
  hr { display: none; }
  .footer-note { margin-top: 8mm; color: var(--muted); font-family: 'DM Mono', monospace; font-size: 8pt; }
"""


def build_html() -> str:
    body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"),
        extensions=["extra", "sane_lists", "smarty"],
    )
    generated = datetime.now(tz=timezone.utc).strftime("%B %d, %Y · %H:%M UTC")
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<base href="{HERE.as_uri()}/">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
  <div class="cover">
    <div class="eyebrow">SherlockML / autonomous ML incident command</div>
    <h1>Demo Walkthrough</h1>
    <div class="sub">An autonomous AI detective and doctor for machine-learning systems —
    a screen-by-screen guide to the live investigation: evidence, debate, diagnosis,
    treatment, and validated recovery.</div>
    <div class="chips">
      <span class="chip">CASE-DRIFT-001 · LIVE RUN</span>
      <span class="chip blue">LANGGRAPH MULTI-AGENT</span>
      <span class="chip blue">FASTAPI + STREAMLIT</span>
      <span class="chip gold">HUMAN APPROVAL REQUIRED</span>
    </div>
    <div class="date">Generated {generated}</div>
  </div>
  <main>
    {body}
    <div class="footer-note">SherlockML is a deterministic, offline-first technical demonstration.
    All data is synthetic; recovery approval is a recommendation for human review, never a deployment.</div>
  </main>
</body>
</html>"""


async def render() -> None:
    from playwright.async_api import async_playwright

    html_path = HERE / "_walkthrough.html"
    html_path.write_text(build_html(), encoding="utf-8")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(html_path.as_uri(), wait_until="networkidle")
            await page.pdf(
                path=str(OUTPUT),
                format="A4",
                print_background=True,
                margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            )
            await browser.close()
    finally:
        html_path.unlink(missing_ok=True)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1_048_576:.1f} MiB)")


if __name__ == "__main__":
    asyncio.run(render())
