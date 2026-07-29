"""Website ingestion: fetch and clean the readable text content of a URL."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from backend.utils.config import settings
from backend.utils.helpers import clean_text
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class WebLoadError(RuntimeError):
    """Raised when a web page cannot be fetched or parsed."""


@dataclass
class WebPageContent:
    """Extracted content from a single web page."""

    url: str
    title: str
    text: str


_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "svg",
    "form",
    "aside",
)


def load_website(url: str) -> WebPageContent:
    """Fetch a URL and extract its main readable text content.

    Args:
        url: A fully qualified HTTP(S) URL.

    Returns:
        A :class:`WebPageContent` instance with the page title and text.

    Raises:
        WebLoadError: On invalid URLs, network failures, or non-HTML content.
    """

    import requests
    from bs4 import BeautifulSoup
    from requests.exceptions import RequestException

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise WebLoadError(f"Invalid URL: '{url}'. Must start with http(s)://")

    headers = {"User-Agent": settings.web_user_agent}

    try:
        response = requests.get(
            url, headers=headers, timeout=settings.web_request_timeout
        )
        response.raise_for_status()
    except RequestException as exc:
        raise WebLoadError(f"Failed to fetch '{url}': {exc}") from exc

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type and "text" not in content_type:
        raise WebLoadError(
            f"Unsupported content type '{content_type}' for URL '{url}'."
        )

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as exc:  # noqa: BLE001
        raise WebLoadError(f"Failed to parse HTML from '{url}': {exc}") from exc

    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    main = soup.find("main") or soup.find("article") or soup.body or soup
    raw_text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    text = clean_text(raw_text)

    if not text:
        raise WebLoadError(f"No readable text content found at '{url}'.")

    return WebPageContent(url=url, title=title, text=text)
