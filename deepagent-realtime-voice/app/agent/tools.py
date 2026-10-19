"""Agent tools: web search (DuckDuckGo) and page fetching.

``fetch_page`` only reaches public internet addresses. Pages can carry prompt
injections, and this compute exposes credential endpoints on loopback and
link-local addresses (managed identity, instance metadata), so every hop —
including redirects — is resolved and checked before connecting.
"""

import asyncio
import ipaddress
import re
import socket
from datetime import date
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import tool

from app import config

MAX_REDIRECTS = 3
MAX_DOWNLOAD_BYTES = 2_000_000
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) deepagent-realtime-voice/1.0"


@tool
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo) for current or factual information you don't
    already know: news, weather, prices, sports, recent events, or facts you are
    unsure about. Returns the top results with titles, snippets and URLs."""
    query = query.strip()
    if not query:
        return "No query provided."
    # Never raise: an exception here would abort the whole agent run (and the spoken reply).
    try:
        results = DDGS(timeout=config.SEARCH_TIMEOUT_S).text(query, max_results=config.SEARCH_MAX_RESULTS)
    except DDGSException as exc:  # ddgs raises when there are no results
        results, error = [], str(exc)
    except Exception as exc:
        return f"Web search failed ({type(exc).__name__}). Try a different query."
    else:
        error = ""
    # Date-stamped so the model treats these as fresh facts, not training knowledge.
    lines = [f"Web search results from {date.today().isoformat()} for {query!r}:"]
    for r in results or []:
        lines.append(f"- {r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})")
    return "\n".join(lines) if len(lines) > 1 else f"{lines[0]}\n{error or 'No results found.'}"


@tool
async def fetch_page(url: str) -> str:
    """Read the text of a web page, e.g. a promising web_search result, when the
    snippets don't contain the actual answer (live values, details, figures)."""
    try:
        text = await fetch_text(url.strip())
    except PageFetchError as exc:
        return f"Could not fetch {url}: {exc}"
    except Exception as exc:  # never raise into the agent run
        return f"Could not fetch {url}: {type(exc).__name__}"
    return f"Content of {url} (fetched {date.today().isoformat()}):\n{text}"


class PageFetchError(Exception):
    pass


async def ensure_public_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise PageFetchError("only http(s) URLs are allowed")
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, parts.hostname, parts.port or 443,
                                        type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise PageFetchError(f"cannot resolve {parts.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise PageFetchError("address is not on the public internet")


async def fetch_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=config.FETCH_TIMEOUT_S, follow_redirects=False,
                                 trust_env=config.FETCH_USE_ENV_PROXY,
                                 headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await ensure_public_url(url)
            try:
                async with client.stream("GET", url) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers.get("location", ""))
                        continue
                    if response.status_code != 200:
                        raise PageFetchError(f"HTTP {response.status_code}")
                    ctype = response.headers.get("content-type", "")
                    if not ctype.startswith(("text/html", "text/plain", "application/xhtml")):
                        raise PageFetchError(f"unsupported content type {ctype or 'unknown'}")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) >= MAX_DOWNLOAD_BYTES:
                            break
                    raw = body.decode(response.encoding or "utf-8", errors="replace")
            except httpx.HTTPError as exc:
                raise PageFetchError(type(exc).__name__) from exc
            text = html_to_text(raw) if "html" in ctype else raw
            return text[: config.FETCH_MAX_CHARS] or "(empty page)"
    raise PageFetchError("too many redirects")


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "template"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
             "article", "table", "ul", "ol", "dd", "dt", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Main article text (trafilatura); falls back to plain tag stripping for
    pages it finds no main content in."""
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True,
                                   deduplicate=True)
    except Exception:
        text = None
    return text or strip_html(html)


def strip_html(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(parser.parts))
    return re.sub(r"\s*\n\s*", "\n", text).strip()
