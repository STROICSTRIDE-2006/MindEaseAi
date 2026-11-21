"""Google OAuth and server-side logout routes for MindEase AI."""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from urllib.parse import urlencode

import requests
from starlette.responses import RedirectResponse, PlainTextResponse
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
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
OAUTH_TTL_SECONDS = 10 * 60


def base_url():
    configured = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured
    host = os.getenv("HOST", "127.0.0.1").strip()
    port = os.getenv("PORT", "7860").strip()
    return f"http://{host}:{port}"


def providers() -> dict:
    """Return configured external sign-in providers."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        return {}

    return {
        "google": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
            "scope": "openid email profile",
        }
    }


def enabled_providers() -> list[str]:
    return list(providers().keys())



def _cookie_secure(request):
    forwarded = request.headers.get("x-forwarded-proto", "").lower()
    return request.url.scheme == "https" or forwarded == "https"


def _set_cookie(response, name, value, request, max_age):
    response.set_cookie(
        name,
        value,
        max_age=max_age,
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        path="/",
    )


def _clear_cookie(response, name, request):
    response.delete_cookie(
        name, path="/", httponly=True, secure=_cookie_secure(request), samesite="lax"
    )


def build_auth_routes(create_session, delete_session):
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv(
        "GOOGLE_REDIRECT_URI", f"{base_url()}/auth/google/callback"
    ).strip()

    async def logout(request):
        token = cookie_from_header(request.headers.get("cookie", ""), SESSION_COOKIE)
        if token:
            try:
                delete_session(token)
            except Exception:
                log.exception("Could not delete server-side session")
        response = RedirectResponse(url="/", status_code=303)
        _clear_cookie(response, SESSION_COOKIE, request)
        _clear_cookie(response, OAUTH_COOKIE, request)
        return response

    async def google_login(request):
        if not google_client_id or not google_client_secret:
            return PlainTextResponse("Google sign-in is not configured.", status_code=503)

        ip = client_ip(
            request.client.host if request.client else "unknown",
            request.headers.get("x-forwarded-for", ""),
        )
        if not limiter.allow(f"oauth:{ip}", 10, 300):
            return PlainTextResponse("Too many sign-in attempts. Please try again later.", status_code=429)

        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        signed = sign_token(
            {"state": state, "verifier": verifier},
            "google_oauth",
            OAUTH_TTL_SECONDS,
        )
        params = {
            "client_id": google_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
        response = RedirectResponse(
            "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params),
            status_code=302,
        )
        _set_cookie(response, OAUTH_COOKIE, signed, request, OAUTH_TTL_SECONDS)
        return response

    async def google_callback(request):
        error = request.query_params.get("error")
        if error:
            response = RedirectResponse(url="/?auth=cancelled", status_code=303)
            _clear_cookie(response, OAUTH_COOKIE, request)
            return response

        state = request.query_params.get("state", "")
        code = request.query_params.get("code", "")
        signed = cookie_from_header(request.headers.get("cookie", ""), OAUTH_COOKIE)
        payload = verify_token(signed, "google_oauth")
        if not payload or not secrets.compare_digest(str(payload.get("state", "")), state) or not code:
            response = RedirectResponse(url="/?auth=failed", status_code=303)
            _clear_cookie(response, OAUTH_COOKIE, request)
            return response

        try:
            token_response = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": google_client_id,
                    "client_secret": google_client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                    "code_verifier": payload["verifier"],
                },
                timeout=20,
            )
            token_response.raise_for_status()
            access_token = token_response.json().get("access_token")
            if not access_token:
                raise RuntimeError("Google did not return an access token.")

            profile_response = requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=20,
            )
            profile_response.raise_for_status()
            profile = profile_response.json()
            email = str(profile.get("email", "")).strip().lower()
            verified = profile.get("verified_email") is True
            if not verified or not valid_email(email):
                raise RuntimeError("Google account email is not verified.")

            session_token = create_session(email, "google")
            response = RedirectResponse(url="/", status_code=303)
            _set_cookie(response, SESSION_COOKIE, session_token, request, SESSION_TTL_SECONDS)
            _clear_cookie(response, OAUTH_COOKIE, request)
            return response
        except Exception:
            log.exception("Google OAuth callback failed")
            response = RedirectResponse(url="/?auth=failed", status_code=303)
            _clear_cookie(response, OAUTH_COOKIE, request)
            return response

    return [
        Route("/auth/logout", logout, methods=["GET", "POST"]),
        Route("/auth/google/login", google_login, methods=["GET"]),
        Route("/auth/google/callback", google_callback, methods=["GET"]),
    ]
