"""The static site CI uploads to GitHub Pages."""

from pathlib import Path

from scripts.build_demo import STLITE_VERSION, build_site

ROOT = Path(__file__).resolve().parents[1]


def test_demo_site_bundles_the_app_and_the_snapshot(tmp_path):
    dest = build_site(tmp_path / "site", root=ROOT)
    html = (dest / "index.html").read_text(encoding="utf-8")
    assert f"@stlite/browser@{STLITE_VERSION}" in html
    assert '"theme.primaryColor": "#10B981"' in html
    assert '"theme.base": "dark"' in html
    assert "Static price snapshot" in html
    assert (dest / ".nojekyll").is_file()
    for relative in (
        "app.py",
        "src/covariance.py",
        "src/optimizer.py",
        "data/fetch_data.py",
        "data/snapshot/prices.csv",
        "data/snapshot/snapshot.json",
    ):
        assert (dest / relative).is_file()
        assert f'"{relative}"' in html


def test_pages_workflow_can_be_rerun_by_hand():
    text = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "actions/upload-pages-artifact" in text
    assert "actions/deploy-pages" in text
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/build_demo.py" in ci
