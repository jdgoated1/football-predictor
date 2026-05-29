"""Inject the Plausible analytics snippet into Streamlit's index.html.

Streamlit gives no supported way to add tags to the page <head>, so we patch
the installed package's static index.html at build time. Run this AFTER
`pip install` in the deploy build command:

    pip install -r requirements.txt && python scripts/inject_analytics.py

Controlled by env vars (read at build time):
    PLAUSIBLE_DOMAIN   the site name configured in Plausible, e.g. wcpicks26.app
    PLAUSIBLE_SRC      optional script URL (default https://plausible.io/js/script.js)

If PLAUSIBLE_DOMAIN is unset, the script does nothing (safe no-op), so local
runs and forks without analytics are unaffected.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    domain = os.environ.get("PLAUSIBLE_DOMAIN", "").strip()
    if not domain:
        print("[inject_analytics] PLAUSIBLE_DOMAIN not set — skipping.")
        return 0

    src = os.environ.get("PLAUSIBLE_SRC", "https://plausible.io/js/script.js").strip()
    snippet = f'<script defer data-domain="{domain}" src="{src}"></script>'

    try:
        import streamlit
    except ImportError:
        print("[inject_analytics] streamlit not importable — skipping.", file=sys.stderr)
        return 0

    index_path = Path(streamlit.__file__).parent / "static" / "index.html"
    if not index_path.exists():
        print(f"[inject_analytics] {index_path} not found — skipping.", file=sys.stderr)
        return 0

    html = index_path.read_text(encoding="utf-8")
    if "plausible.io" in html or f'data-domain="{domain}"' in html:
        print("[inject_analytics] Plausible already present — nothing to do.")
        return 0
    if "</head>" not in html:
        print("[inject_analytics] no </head> in index.html — skipping.", file=sys.stderr)
        return 0

    html = html.replace("</head>", snippet + "</head>", 1)
    index_path.write_text(html, encoding="utf-8")
    print(f"[inject_analytics] Injected Plausible for domain '{domain}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
