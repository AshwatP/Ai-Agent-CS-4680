"""
actions/http_api.py — Outbound HTTP GET to an allowlist of public APIs.

Uses httpx (installed with the anthropic SDK). Redirects are NOT followed,
since a redirect could point outside the allowlist.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

import httpx

from actions.base import Action, ActionContext
from models import ActionResult, RiskLevel
from safety import SafetyError

ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "api.open-meteo.com",
        "geocoding-api.open-meteo.com",
        "api.github.com",
        "catfact.ninja",
        "official-joke-api.appspot.com",
    }
)

_USER_AGENT = "llm-action-agent/1.0 (course project)"

# Subset of WMO weather codes used by Open-Meteo.
_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "rain showers", 81: "heavy rain showers", 95: "thunderstorm",
}


def check_url(url: str) -> None:
    """Allow only https URLs on allowlisted hosts, default port, no credentials."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise SafetyError("Only https:// URLs are allowed.")
    if parts.username or parts.password:
        raise SafetyError("URLs with embedded credentials are not allowed.")
    if parts.port not in (None, 443):
        raise SafetyError("Non-standard ports are not allowed.")
    host = (parts.hostname or "").lower()
    if host not in ALLOWED_DOMAINS:
        raise SafetyError(
            f"Domain '{host}' is not on the allowlist. Allowed: {', '.join(sorted(ALLOWED_DOMAINS))}."
        )


def _client(ctx: ActionContext) -> httpx.Client:
    return httpx.Client(
        timeout=ctx.http_timeout,
        transport=ctx.http_transport,   # None in production, MockTransport in tests
        follow_redirects=False,
        headers={"User-Agent": _USER_AGENT},
    )


class HttpGet(Action):
    name = "http_get"
    description = (
        "Send an HTTP GET request to an allowlisted public API and return the response body. "
        f"Allowed domains: {', '.join(sorted(ALLOWED_DOMAINS))}."
    )
    risk = RiskLevel.SAFE
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "minLength": 8, "maxLength": 2000},
            "query": {
                "type": "object",
                "description": "Optional query-string parameters.",
                "additionalProperties": {"type": ["string", "number", "boolean"]},
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def safety_check(self, p: dict[str, Any], ctx: ActionContext) -> str | None:
        check_url(p["url"])
        return None

    def describe(self, p: dict[str, Any], ctx: ActionContext) -> str:
        return f"GET {p['url']}" + (f" ?{p['query']}" if p.get("query") else "")

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        check_url(p["url"])
        try:
            with _client(ctx) as client:
                resp = client.get(p["url"], params=p.get("query"))
        except httpx.HTTPError as exc:
            return ActionResult.fail(f"Network error: {exc}")

        body = resp.text
        if "json" in resp.headers.get("content-type", ""):
            try:
                body = json.dumps(resp.json(), indent=2)
            except ValueError:
                pass
        summary = f"HTTP {resp.status_code}\n{body}"
        if resp.is_success:
            return ActionResult.ok(summary, status=resp.status_code)
        return ActionResult(success=False, output=summary, error=summary,
                            data={"status": resp.status_code})


class GetWeather(Action):
    name = "get_weather"
    description = "Get the current weather for a city (uses the free Open-Meteo API)."
    risk = RiskLevel.SAFE
    input_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1, "maxLength": 100,
                     "description": "City name, e.g. 'San Francisco'."}
        },
        "required": ["city"],
        "additionalProperties": False,
    }

    def describe(self, p: dict[str, Any], ctx: ActionContext) -> str:
        return f"GET current weather for {p['city']}"

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        try:
            with _client(ctx) as client:
                geo = client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": p["city"], "count": 1},
                )
                geo.raise_for_status()
                results = geo.json().get("results") or []
                if not results:
                    return ActionResult.fail(f"Could not find a city named '{p['city']}'.")
                place = results[0]

                wx = client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": place["latitude"],
                        "longitude": place["longitude"],
                        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                        "timezone": "auto",
                    },
                )
                wx.raise_for_status()
                cur = wx.json()["current"]
        except httpx.HTTPError as exc:
            return ActionResult.fail(f"Weather API error: {exc}")
        except (KeyError, ValueError) as exc:
            return ActionResult.fail(f"Unexpected weather API response: {exc}")

        conditions = _WEATHER_CODES.get(cur.get("weather_code"), "unknown conditions")
        where = ", ".join(x for x in (place.get("name"), place.get("admin1"), place.get("country")) if x)
        text = (
            f"Current weather in {where}: {conditions}, "
            f"{cur['temperature_2m']}°C, humidity {cur['relative_humidity_2m']}%, "
            f"wind {cur['wind_speed_10m']} km/h (as of {cur.get('time', 'now')} local)."
        )
        return ActionResult.ok(text, temperature_c=cur["temperature_2m"], location=where)
