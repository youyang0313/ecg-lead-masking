"""Render PRESENTATION_SCRIPT.md to PRESENTATION_SCRIPT.pdf (Markdown -> HTML -> Edge headless print).

Usage: python make_script_pdf.py
"""
import re
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 18mm 18mm 18mm 18mm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt 0; padding-bottom: 3pt; border-bottom: 1px solid #c8c8c8; page-break-after: avoid; }
p { margin: 4pt 0 7pt 0; }
p.cue { margin: 12pt 0 3pt 0; padding: 3pt 8pt; background: #eef3fb; border-left: 3px solid #2b6cb0;
        font-size: 9.5pt; color: #2b4a75; page-break-after: avoid; }
code { font-family: Consolas, monospace; font-size: 9.5pt; background: #f2f2f2; padding: 0 2pt; border-radius: 2px; }
hr { border: none; margin: 6pt 0; }
ul { margin: 4pt 0 0 0; padding-left: 16pt; }
li { margin-bottom: 6pt; }
.meta { color: #555; font-size: 10pt; }
"""


def main():
    md_text = (HERE/"PRESENTATION_SCRIPT.md").read_text(encoding="utf-8")
    html = markdown.markdown(md_text, extensions=["extra"])
    html = re.sub(r"<p>▶ ", '<p class="cue">▶ ', html)              # click cues stand out from the spoken text
    html = html.replace("<hr />", "")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Presentation script</title>
<style>{CSS}</style></head><body>{html}</body></html>"""
    out_html = HERE/"PRESENTATION_SCRIPT.html"
    out_html.write_text(page, encoding="utf-8")
    out_pdf = HERE/"PRESENTATION_SCRIPT.pdf"
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, timeout=120)
    print("wrote", out_pdf)


if __name__ == "__main__":
    main()
