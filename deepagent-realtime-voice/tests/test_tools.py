"""Agent tools: fetch_page safety + extraction (always), live fetch (AGENT_TESTS=1)."""

import asyncio
import os

import pytest

from app.agent import tools
from app.agent.tools import (PageFetchError, ensure_public_url, fetch_page, fetch_text, html_to_text,
                             strip_html, web_search)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:46808/MSI/auth",          # managed identity endpoint on this compute
    "http://169.254.169.254/metadata/instance",  # cloud instance metadata
    "http://localhost:7860/api/config",
    "http://10.0.0.5/",
    "http://[::1]/",
    "file:///etc/passwd",
    "ftp://example.com/",
    "not a url",
])
def test_fetch_refuses_non_public_targets(url):
    with pytest.raises(PageFetchError):
        asyncio.run(ensure_public_url(url))


def test_fetch_page_reports_refusals_to_the_model_instead_of_raising():
    out = asyncio.run(fetch_page.ainvoke({"url": "http://169.254.169.254/latest/meta-data/"}))
    assert out.startswith("Could not fetch") and "public internet" in out


def test_web_search_reports_failures_instead_of_raising(monkeypatch):
    """An exception escaping a tool aborts the whole agent run (and the spoken reply)."""
    from ddgs.exceptions import DDGSException

    class NoResults:
        def __init__(self, **kwargs): pass
        def text(self, *args, **kwargs): raise DDGSException("No results found.")

    class Broken(NoResults):
        def text(self, *args, **kwargs): raise ConnectionError("network down")

    monkeypatch.setattr(tools, "DDGS", NoResults)
    assert web_search.invoke({"query": "zzqx"}).endswith("No results found.")
    monkeypatch.setattr(tools, "DDGS", Broken)
    assert web_search.invoke({"query": "zzqx"}).startswith("Web search failed (ConnectionError)")


def test_strip_html_keeps_content_and_drops_chrome():
    html = """<html><head><title>T</title><style>p{}</style></head><body>
      <nav>Menu Home</nav><h1>Paris weather</h1><p>Now: 21&deg;C,&nbsp;sunny.</p>
      <script>track()</script><ul><li>Wind 10 km/h</li></ul><footer>(c) site</footer></body></html>"""
    assert strip_html(html) == "Paris weather\nNow: 21°C,\xa0sunny.\nWind 10 km/h"


def test_html_to_text_prefers_the_main_article():
    article = " ".join(["Paris is the capital of France and its largest city."] * 20)
    html = (f"<html><body><nav><ul>{'<li>Language</li>' * 200}</ul></nav>"
            f"<article><h1>Paris</h1><p>{article}</p></article></body></html>")
    text = html_to_text(html)
    assert text.startswith(("Paris", "# Paris")) and "capital of France" in text
    assert "Language" not in text


@pytest.mark.skipif(os.environ.get("AGENT_TESTS") != "1", reason="set AGENT_TESTS=1 for internet access")
def test_live_fetch_reads_public_pages():
    assert "documentation examples" in asyncio.run(fetch_text("https://example.com/"))
    wiki = asyncio.run(fetch_text("https://en.wikipedia.org/wiki/Paris"))
    assert "capital" in wiki[:1500] and "Afrikaans" not in wiki[:1500]
