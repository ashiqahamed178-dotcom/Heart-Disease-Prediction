"""Capture README screenshots from a running app (dev only).

pip install playwright
streamlit run app/streamlit_app.py --server.port 8599
python scripts/screenshots.py http://localhost:8599
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def main(base: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        # Streamlit scrolls inside its own container, so size the viewport to the page.
        for name, path, height in (("predict", "", 1300), ("model", "model", 2600)):
            page = browser.new_page(viewport={"width": 1400, "height": height})
            page.goto(f"{base}/{path}")
            page.wait_for_selector("text=Not for medical use", timeout=60_000)
            page.wait_for_load_state("networkidle")
            if name == "predict":
                page.wait_for_selector("text=Why this estimate", timeout=60_000)
            page.wait_for_timeout(3000)
            page.screenshot(path=OUT / f"{name}.png")
            print("saved", OUT / f"{name}.png")
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8599")
