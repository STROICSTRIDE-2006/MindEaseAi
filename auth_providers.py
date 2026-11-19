"""MindEase AI authentication providers: Google OAuth only.

Only Google OAuth is handled here. Email OTP remains implemented in
app.py and is not handled by this module.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from urllib.parse import urlencode

import requests
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

from security import (
    client_ip,
    cookie_from_header,
    limiter,
    sign_token,
    valid_email,
    verify_token,
)

log = logging.getLogger("mindease.auth")
SESSION_COOKIE = "me_session"
OAUTH_COOKIE = "me_oauth"
SESSION_TTL = 7 * 24 * 60 * 60


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def base_url() -> str:
    """Public site URL; set PUBLIC_BASE_URL to the exact HTTPS origin."""
    explicit = _env("PUBLIC_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    return f"http://{_env('HOST') or '127.0.0.1'}:{_env('PORT') or '7860'}"


def providers() -> dict:
    """Return configured external sign-in providers (Google only)."""
    if not (_env("GOOGLE_CLIENT_ID") and _env("GOOGLE_CLIENT_SECRET")):
        return {}
    return {
        "google": {
            "client_id": _env("GOOGLE_CLIENT_ID"),
            "client_secret": _env("GOOGLE_CLIENT_SECRET"),
            "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
            "scope": "openid email",
        }
    }


def enabled_providers() -> list[str]:
    return list(providers().keys())


def _ip(request) -> str:
    peer = request.client.host if request.client else "unknown"
    return client_ip(peer, request.headers.get("x-forwarded-for", ""))


def _secure(request) -> bool:
    return base_url().startswith("https://") or request.headers.get(
        "x-forwarded-proto", ""
    ).lower() == "https"


def _no_store(response):
    response.headers["Cache-Control"] = "no-store"
    return response


def _failed(status: int = 400):
    page = (
        "<!doctype html><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Sign-in failed</title>"
        "<body style='font-family:Inter,Arial,sans-serif;max-width:460px;"
        "margin:12vh auto;padding:0 20px;color:#203b55'>"
        "<h2>Sign-in didn’t work</h2>"
        "<p>Something went wrong while signing you in. Please go back and try again.</p>"
        "<p><a href='/' style='color:#286b9b;font-weight:700'>"
        "Back to MindEase AI</a></p></body>"
    )
    return _no_store(HTMLResponse(page, status_code=status))


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _make_login():
    def login(request):
        cfg = providers().get("google")
        if not cfg:
            return _failed(404)
        if not limiter.allow(f"oauth:{_ip(request)}", 20, 600):
            return _failed(429)

        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        params = {
            "client_id": cfg["client_id"],
            "redirect_uri": f"{base_url()}/auth/google/callback",
            "response_type": "code",
            "scope": cfg["scope"],
            "state": state,
            "code_challenge": _pkce_challenge(verifier),
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
        response = RedirectResponse(
            f"{cfg['auth_url']}?{urlencode(params)}", status_code=303
        )
        response.set_cookie(
            OAUTH_COOKIE,
            sign_token({"s": state, "v": verifier, "p": "google"}, "oauth", 600),
            max_age=600,
            httponly=True,
            samesite="lax",
            secure=_secure(request),
            path="/auth",
        )
        return _no_store(response)

    return login


def _make_callback(create_session):
    def callback(request):
        cfg = providers().get("google")
        if not cfg:
            return _failed(404)
        if not limiter.allow(f"oauth:{_ip(request)}", 20, 600):
            return _failed(429)

        query = request.query_params
        if query.get("error"):
            return _failed()
        code = query.get("code", "")
        state = query.get("state", "")
        saved = verify_token(
            cookie_from_header(request.headers.get("cookie", ""), OAUTH_COOKIE),
            "oauth",
        )
        if (
            not saved
            or saved.get("p") != "google"
            or not code
            or len(code) > 4096
            or not state
            or not hmac.compare_digest(state, str(saved.get("s", "")))
        ):
            return _failed()

        try:
            token_response = requests.post(
                cfg["token_url"],
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": f"{base_url()}/auth/google/callback",
                    "client_id": cfg["client_id"],
                    "client_secret": cfg["client_secret"],
                    "code_verifier": saved.get("v", ""),
                },
                headers={"Accept": "application/json"},
                timeout=15,
            )
            token_response.raise_for_status()
            access_token = token_response.json().get("access_token")
            if not access_token:
                raise RuntimeError("Google did not return an access token")

            info_response = requests.get(
                cfg["userinfo_url"],
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15,
            )
            info_response.raise_for_status()
            info = info_response.json()
        except Exception as error:
            log.error("Google sign-in failed: %r", error)
            return _failed()

        email = str(info.get("email") or "").strip().lower()
        if info.get("email_verified") not in (True, "true") or not valid_email(email):
            log.warning("Google account without a verified valid e-mail was refused.")
            return _failed()

        session_token = create_session(email, "google")
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            session_token,
            max_age=SESSION_TTL,
            httponly=True,
            samesite="lax",
            secure=_secure(request),
            path="/",
        )
        response.delete_cookie(OAUTH_COOKIE, path="/auth")
        return _no_store(response)

    return callback


def _make_logout(delete_session):
    def logout(request):
        token = cookie_from_header(request.headers.get("cookie", ""), SESSION_COOKIE)
        if token:
            try:
                delete_session(token)
            except Exception as error:
                log.error("Could not delete session: %r", error)
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return _no_store(response)

    return logout


def build_auth_routes(create_session, delete_session):
    """Build Google OAuth and logout routes. Email OTP is handled in app.py."""
    return [
        Route("/auth/logout", _make_logout(delete_session), methods=["POST"]),
        Route("/auth/google/login", _make_login(), methods=["GET"]),
        Route(
            "/auth/google/callback",
            _make_callback(create_session),
            methods=["GET"],
        ),
    ]
