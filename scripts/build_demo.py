"""Assemble the static GitHub Pages site that boots the Streamlit app.

The site is stlite (Streamlit on Pyodide). It copies this repo's app and the
bundled price snapshot; it does not call Yahoo Finance. Output defaults to
``site/``, which CI uploads with actions/upload-pages-artifact.

    python scripts/build_demo.py
    python scripts/build_demo.py --out /tmp/portfolio-demo
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STLITE_VERSION = "1.9.2"
DEMO_FILES = (
    "app.py",
    "src/__init__.py",
    "src/backtest.py",
    "src/data_handler.py",
    "src/metrics.py",
    "src/optimizer.py",
    "src/visualization.py",
    "data/__init__.py",
    "data/fetch_data.py",
    "data/snapshot/prices.csv",
    "data/snapshot/snapshot.json",
)


def _snapshot_blurb(root: Path) -> str:
    meta = json.loads((root / "data" / "snapshot" / "snapshot.json").read_text(encoding="utf-8"))
    tickers = meta["tickers"]
    names = ", ".join(tickers[:-1]) + f", and {tickers[-1]}"
    return (
        "Static price snapshot, not live Yahoo Finance. "
        f"Split- and dividend-adjusted closes for {names}, "
        f"from {meta['first_session']} through {meta['last_session']} "
        f"(downloaded {meta['downloaded_on']})."
    )


def _validate_snapshot(root: Path) -> None:
    meta_path = root / "data" / "snapshot" / "snapshot.json"
    prices_path = root / "data" / "snapshot" / "prices.csv"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with prices_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3:
        raise SystemExit(f"{prices_path} does not contain a price history")
    header, first, last = rows[0], rows[1], rows[-1]
    if header[1:] != list(meta["tickers"]):
        raise SystemExit(
            f"Snapshot columns {header[1:]} do not match metadata tickers {meta['tickers']}"
        )
    if first[0] != meta["first_session"] or last[0] != meta["last_session"]:
        raise SystemExit(
            "Snapshot date range "
            f"{first[0]} → {last[0]} does not match metadata "
            f"{meta['first_session']} → {meta['last_session']}"
        )


def _index_html(file_map: dict[str, str], blurb: str) -> str:
    files_js = ",\n          ".join(
        f'"{path}": {{ url: "./{url}" }}' for path, url in file_map.items()
    )
    # The blurb is plain text from our own snapshot metadata (tickers and dates).
    safe_blurb = (
        blurb.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <title>Markowitz Portfolio Optimizer</title>
    <meta
      name="description"
      content="In-browser Markowitz portfolio optimizer. Prices are a static Yahoo Finance snapshot."
    />
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build/stlite.css"
    />
    <style>
      html, body, #root {{
        margin: 0;
        min-height: 100%;
        background: #0B1220;
        color: #E5E7EB;
        font-family: "Source Sans Pro", "Segoe UI", sans-serif;
      }}
      .boot {{
        max-width: 40rem;
        padding: 3rem 1.5rem;
      }}
      .boot h1 {{
        margin: 0 0 0.75rem;
        color: #10B981;
        font-size: 1.6rem;
        font-weight: 650;
      }}
      .boot p {{
        margin: 0.4rem 0;
        line-height: 1.5;
      }}
      .boot .muted {{
        color: #9CA3AF;
        font-size: 0.95rem;
      }}
    </style>
  </head>
  <body>
    <div id="root">
      <div class="boot">
        <h1>Markowitz Portfolio Optimizer</h1>
        <p>{safe_blurb}</p>
        <p class="muted">
          Loading the Streamlit app in your browser. The first visit downloads
          the Python runtime; nothing is installed on this computer, and no
          server is kept running. Then use Optimise in the sidebar.
        </p>
      </div>
    </div>
    <script type="module">
      import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build/stlite.js";
      mount(
        {{
          entrypoint: "app.py",
          requirements: ["numpy", "pandas", "scipy", "plotly", "jinja2"],
          files: {{
          {files_js}
          }},
          streamlitConfig: {{
            "theme.base": "dark",
            "theme.primaryColor": "#10B981",
            "theme.backgroundColor": "#0B1220",
            "theme.secondaryBackgroundColor": "#111827",
            "theme.textColor": "#E5E7EB",
            "client.toolbarMode": "viewer"
          }}
        }},
        document.getElementById("root"),
      );
    </script>
  </body>
</html>
"""


def build_site(dest: Path, root: Path = ROOT) -> Path:
    """Copy the app into ``dest`` and write index.html. Raises SystemExit on failure."""
    _validate_snapshot(root)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    file_map: dict[str, str] = {}
    for relative in DEMO_FILES:
        source = root / relative
        if not source.is_file():
            raise SystemExit(f"Demo build is missing {relative}")
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        file_map[relative] = relative
    blurb = _snapshot_blurb(root)
    html = _index_html(file_map, blurb)
    required = (
        f"@stlite/browser@{STLITE_VERSION}",
        "#10B981",
        "theme.base",
        "Static price snapshot",
        "data/snapshot/prices.csv",
        "app.py",
    )
    missing = [token for token in required if token not in html]
    if missing:
        raise SystemExit(f"Demo index.html is missing {missing}")
    (dest / "index.html").write_text(html, encoding="utf-8")
    (dest / ".nojekyll").write_text("", encoding="utf-8")
    _validate_snapshot(dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the GitHub Pages demo site.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "site",
        help="Directory to write (default: site/)",
    )
    args = parser.parse_args(argv)
    dest = build_site(args.out.resolve())
    print(f"Built {dest}")


if __name__ == "__main__":
    main(sys.argv[1:])
