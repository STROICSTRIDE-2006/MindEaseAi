"""
security.py - security helpers for MindEase AI.

Put this file next to your app file and import from it.
Contains: input cleaning, markdown/URL sanitising, rate limiting,
and a pure-ASGI middleware (security headers, host check, body limit,
per-IP request limits, upload blocking).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import http.cookies
import json
import logging
import os
import re
import threading
import time
from collections import deque
from urllib.parse import quote, urlparse

log = logging.getLogger("mindease.security")

# ---------------------------------------------------------------
# Limits (change here, one place)
# ---------------------------------------------------------------
MAX_THOUGHTS = 4000
MAX_QUERY = 300
MAX_TITLE = 200
MAX_NOTES = 1000
MAX_REMINDERS_PER_USER = 200
ALLOWED_LEADS = {"0", "5", "15", "30"}
DAILY_AI_CAP = int(os.getenv("DAILY_AI_CAP", "500"))  # all users together

# ---------------------------------------------------------------
# Input cleaning
# ---------------------------------------------------------------
# control characters + invisible bidi-override characters
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u202a-\u202e\u2066-\u2069]")


def clean_text(value, max_len, *, multiline=True):
    """Strip control chars, trim, and enforce a length limit."""
    text = "" if value is None else str(value)
    text = _CTRL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    if not multiline:
        text = " ".join(text.split())
    text = text.strip()
    if len(text) > max_len:
        raise ValueError(f"Text is too long (maximum {max_len} characters).")
    return text


def md_escape(value, max_len=500):
    """Make untrusted text safe to place inside Markdown."""
    text = _CTRL.sub("", str(value or ""))[:max_len]
    text = html.escape(text, quote=True)  # neutralises raw HTML tags
    return re.sub(r"([\\`*_{}\[\]()#+\-!|~$])", r"\\\1", text)


def safe_link_url(url):
    """Return a safe http(s) URL for a Markdown link, or '' if unsafe."""
    if not isinstance(url, str):
        return ""
    url = html.unescape(url).strip().replace(" ", "%20")
    if not url or len(url) > 2000 or re.search(r"[\s\x00-\x1f\x7f]", url):
        return ""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except ValueError:
        return ""
    if parsed.scheme not in ("http", "https") or not host:
        return ""
    if parsed.username or parsed.password:
        return ""
    # quote() also encodes ( ) ' " < > ` so a URL cannot break out of the link
    return quote(url, safe=":/?#[]@!$&*+,;=%-._~")


# ---------------------------------------------------------------
# E-mail check, signed tokens, cookies
# ---------------------------------------------------------------
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)


def valid_email(value):
    value = (value or "").strip()
    return len(value) <= 254 and bool(_EMAIL_RE.fullmatch(value))


def _b64e(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _key(purpose):
    secret = os.getenv("OTP_SECRET", "").strip()
    if len(secret) < 32:
        raise RuntimeError("OTP_SECRET is missing or too short.")
    return hmac.new(secret.encode("utf-8"), b"mindease:" + purpose.encode("utf-8"),
                    hashlib.sha256).digest()


def sign_token(payload, purpose, ttl):
    """Tamper-proof, expiring token (HMAC-SHA256). Used for the OAuth state cookie."""
    body = dict(payload)
    body["pur"] = purpose
    body["exp"] = int(time.time()) + int(ttl)
    raw = _b64e(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    sig = _b64e(hmac.new(_key(purpose), raw.encode("ascii"), hashlib.sha256).digest())
    return f"{raw}.{sig}"


def verify_token(token, purpose):
    """Return the payload if the signature, purpose and expiry are valid, else None."""
    try:
        raw, sig = str(token or "").split(".", 1)
        expected = _b64e(hmac.new(_key(purpose), raw.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        body = json.loads(_b64d(raw))
        if body.get("pur") != purpose or int(body.get("exp", 0)) < time.time():
            return None
        return body
    except Exception:
        return None


def cookie_from_header(cookie_header, name):
    try:
        jar = http.cookies.SimpleCookie()
        jar.load(cookie_header or "")
        morsel = jar.get(name)
        return morsel.value if morsel else ""
    except Exception:
        return ""


# ---------------------------------------------------------------
# Rate limiting (in-memory, thread-safe)
# ---------------------------------------------------------------
class RateLimiter:
    def __init__(self):
        self._hits = {}
        self._lock = threading.Lock()
        self._last_gc = time.monotonic()

    def allow(self, key, limit, window):
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if now - self._last_gc > 300:
                self._last_gc = now
                for k in [k for k, d in self._hits.items()
                          if not d or now - d[-1] > 86400]:
                    del self._hits[k]
            return True


limiter = RateLimiter()

TRUST_PROXY = os.getenv("TRUST_PROXY", "0").strip() == "1"
try:
    PROXY_HOPS = max(1, int(os.getenv("TRUSTED_PROXY_HOPS", "1")))
except ValueError:
    PROXY_HOPS = 1


def client_ip(peer, forwarded_for=""):
    """X-Forwarded-For is only trusted when TRUST_PROXY=1 (set it behind nginx etc.)."""
    if TRUST_PROXY and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if len(parts) >= PROXY_HOPS:
            return parts[-PROXY_HOPS][:64]
    return str(peer or "unknown")[:64]


def gradio_ip(request):
    try:
        return client_ip(request.client.host, request.headers.get("x-forwarded-for", ""))
    except Exception:
        return "unknown"


def ai_limit_message(request, bucket, limit, window):
    """Return an error message if this caller is over the limit, else None."""
    ip = gradio_ip(request)
    if not limiter.allow(f"{bucket}:{ip}", limit, window):
        return "You're going a little fast. Please wait a few minutes and try again."
    if not limiter.allow("global:ai", DAILY_AI_CAP, 86400):
        return "The daily AI limit has been reached. Please try again tomorrow."
    return None


# ---------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------
_CSP = (
    "default-src 'self'; "
    # 'unsafe-inline' = your inline HEAD script; 'unsafe-eval' = Gradio's gr.HTML
    # component evaluates JS strings internally (without it every HTML block fails).
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' data: https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "media-src 'self' data: blob:; "
    "connect-src 'self'; "
    "worker-src 'self' blob:; "
    "frame-src 'none'; object-src 'none'; "
    "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)


def _host_only(host):
    host = host.strip().lower()
    if host.startswith("["):
        return host.split("]")[0] + "]"
    return host.split(":")[0]


class SecurityMiddleware:
    def __init__(self, app):
        self.app = app
        self.allowed_hosts = {
            _host_only(h) for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()
        }
        self.csp_mode = os.getenv("CSP_MODE", "enforce").strip().lower()
        try:
            self.max_body = int(os.getenv("MAX_BODY_BYTES", str(1024 * 1024)))
        except ValueError:
            self.max_body = 1024 * 1024

    def _security_headers(self, https):
        items = [
            ("x-content-type-options", "nosniff"),
            ("x-frame-options", "DENY"),
            ("referrer-policy", "strict-origin-when-cross-origin"),
            ("permissions-policy",
             "camera=(), microphone=(), geolocation=(), payment=(), usb=()"),
            ("cross-origin-opener-policy", "same-origin"),
            ("cross-origin-resource-policy", "same-origin"),
        ]
        if self.csp_mode == "enforce":
            items.append(("content-security-policy", _CSP))
        elif self.csp_mode == "report-only":
            items.append(("content-security-policy-report-only", _CSP))
        if https:
            items.append(("strict-transport-security", "max-age=31536000; includeSubDomains"))
        return items

    async def _reject(self, send, status, text, https, extra=()):
        body = text.encode("utf-8")
        headers = [(b"content-type", b"text/plain; charset=utf-8"),
                   (b"content-length", str(len(body)).encode())]
        headers += [(k.encode(), v.encode()) for k, v in list(extra) + self._security_headers(https)]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        h = {k.decode("latin-1").lower(): v.decode("latin-1")
             for k, v in scope.get("headers", [])}
        https = scope.get("scheme") in ("https", "wss") or \
            h.get("x-forwarded-proto", "").lower() == "https"

        if self.allowed_hosts and _host_only(h.get("host", "")) not in self.allowed_hosts:
            if scope["type"] == "http":
                await self._reject(send, 400, "Invalid host.", https)
            else:
                await send({"type": "websocket.close", "code": 1008})
            return

        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET").upper()
        path = scope.get("path", "")

        try:
            too_big = int(h.get("content-length", "0")) > self.max_body
        except ValueError:
            too_big = True
        if too_big:
            await self._reject(send, 413, "Request too large.", https)
            return

        # This app has no file uploads, so close that door completely.
        if method == "POST" and "/upload" in path:
            await self._reject(send, 403, "Uploads are disabled.", https)
            return

        peer = (scope.get("client") or ("unknown", 0))[0]
        ip = client_ip(peer, h.get("x-forwarded-for", ""))

        is_api = method == "POST" and any(
            s in path for s in ("/queue/join", "/run/", "/call/", "/api/"))
        ok = limiter.allow(f"http:{ip}", 600, 60)
        if ok and is_api:
            ok = limiter.allow(f"api:{ip}", 40, 60)
        if not ok:
            await self._reject(send, 429, "Too many requests. Please slow down.",
                               https, extra=[("retry-after", "60")])
            return

        sec = self._security_headers(https)
        names = {k for k, _ in sec}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.decode("latin-1").lower() not in names]
                headers += [(k.encode(), v.encode()) for k, v in sec]
                message = {**message, "headers": headers}
            await send(message)

        started = False

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send_wrapper(message)

        try:
            await self.app(scope, receive, tracked_send)
        except Exception:
            # Gradio runs Starlette in debug mode, which would show a traceback page.
            log.exception("Unhandled server error")
            if started:
                raise
            await self._reject(send, 500, "Something went wrong.", https)