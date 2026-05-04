"""Web fetch sandbox: SSRF defense + allow/deny lists."""
from __future__ import annotations

import pytest

from tools import web_fetch


def test_loopback_refused():
    with pytest.raises(PermissionError, match="private"):
        web_fetch.run({"url": "http://127.0.0.1/"})


def test_private_range_refused():
    with pytest.raises(PermissionError, match="private"):
        web_fetch.run({"url": "http://192.168.1.1/"})


def test_allow_private_override(monkeypatch):
    monkeypatch.setenv("CRYPT_WEB_ALLOW_PRIVATE", "1")
    # Refused later by network or another check, but NOT by the SSRF guard.
    with pytest.raises(Exception) as exc_info:
        web_fetch.run({"url": "http://127.0.0.1:1/"})
    assert "private" not in str(exc_info.value).lower()


def test_denylist(monkeypatch):
    monkeypatch.setenv("CRYPT_WEB_DENIED_HOSTS", "blocked.example")
    with pytest.raises(PermissionError, match="DENIED"):
        web_fetch.run({"url": "https://blocked.example/path"})


def test_allowlist_filters_other_hosts(monkeypatch):
    monkeypatch.setenv("CRYPT_WEB_ALLOWED_HOSTS", "ok.example")
    with pytest.raises(PermissionError, match="not in"):
        web_fetch.run({"url": "https://other.example/"})


def test_host_glob_match():
    assert web_fetch._host_matches("api.github.com", ["*.github.com"])
    assert not web_fetch._host_matches("github.com", ["*.github.com"])
    assert web_fetch._host_matches("github.com", ["github.com"])


def test_non_http_scheme_refused():
    with pytest.raises(ValueError, match="http"):
        web_fetch.run({"url": "file:///etc/passwd"})


def test_empty_url_refused():
    with pytest.raises(ValueError):
        web_fetch.run({"url": ""})
