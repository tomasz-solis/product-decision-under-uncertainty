"""Wake or verify a Streamlit Community Cloud app from GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from http.cookiejar import CookieJar
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener

DEFAULT_USER_AGENT = "github-actions-streamlit-keepalive"
RUNNING_STATUS = 5
SHUTDOWN_STATUS = 12
BOOTING_STATUSES = {0, 1, 2, 3, 4, 6, 7}
STATUS_NAMES = {
    0: "UNKNOWN",
    1: "CREATING",
    2: "CREATED",
    3: "UPDATING",
    4: "INSTALLING",
    5: "RUNNING",
    6: "RESTARTING",
    7: "REBOOTING",
    8: "DELETING",
    9: "DELETED",
    10: "USER_ERROR",
    11: "PLATFORM_ERROR",
    12: "IS_SHUTDOWN",
    13: "INSTALLER_ERROR",
    14: "USER_SCRIPT_ERROR",
    15: "POTENTIAL_MINER_DETECTED",
}


@dataclass(frozen=True)
class StatusResponse:
    """One response from Streamlit Cloud's app status endpoint."""

    http_code: int
    payload: dict[str, Any]
    csrf_token: str | None
    url: str

    @property
    def app_status(self) -> int | None:
        """Return the numeric Streamlit app status if the response includes one."""

        status = self.payload.get("status")
        if isinstance(status, int):
            return status
        return None


class StreamlitKeepaliveError(RuntimeError):
    """Raised when the keepalive check cannot prove the app is running."""


def normalize_app_url(raw_url: str) -> str:
    """Return a Streamlit app URL with a scheme and no trailing slash."""

    app_url = raw_url.strip()
    if not app_url:
        raise StreamlitKeepaliveError("Missing repository variable STREAMLIT_APP_URL.")
    if not app_url.startswith(("http://", "https://")):
        app_url = f"https://{app_url}"

    parsed = urlparse(app_url)
    if not parsed.scheme or not parsed.netloc:
        raise StreamlitKeepaliveError(f"STREAMLIT_APP_URL is not a valid URL: {raw_url!r}")
    return app_url.rstrip("/")


def api_url(app_url: str, path: str) -> str:
    """Build an absolute Streamlit Cloud API URL for the deployed app."""

    return urljoin(f"{app_url}/", path.lstrip("/"))


def status_label(status: int | None) -> str:
    """Return a readable name for a Streamlit Cloud app status."""

    if status is None:
        return "missing"
    return STATUS_NAMES.get(status, f"unknown:{status}")


def read_json_response(response: Any) -> dict[str, Any]:
    """Read a JSON response body, allowing empty bodies from POST endpoints."""

    body = response.read().decode("utf-8").strip()
    if not body:
        return {}
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise StreamlitKeepaliveError("Streamlit Cloud returned JSON that was not an object.")
    return parsed


def request_json(
    opener: Any,
    url: str,
    *,
    method: str = "GET",
    csrf_token: str | None = None,
    timeout_seconds: int,
) -> tuple[int, dict[str, Any], str | None]:
    """Send one JSON-oriented request to Streamlit Cloud."""

    headers = {
        "Accept": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if csrf_token:
        headers["x-csrf-token"] = csrf_token

    request = Request(url, headers=headers, method=method)
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            payload = read_json_response(response)
            return response.status, payload, response.headers.get("x-csrf-token")
    except HTTPError as error:
        try:
            payload = read_json_response(error)
        except (json.JSONDecodeError, StreamlitKeepaliveError):
            payload = {"error": error.reason}
        return error.code, payload, error.headers.get("x-csrf-token")
    except URLError as error:
        raise StreamlitKeepaliveError(f"Could not reach Streamlit Cloud: {error.reason}") from error


def fetch_status(opener: Any, app_url: str, timeout_seconds: int) -> StatusResponse:
    """Fetch the Streamlit Cloud status for a subdomain-hosted app."""

    url = api_url(app_url, "/api/v2/app/status")
    http_code, payload, csrf_token = request_json(
        opener,
        url,
        timeout_seconds=timeout_seconds,
    )
    return StatusResponse(http_code=http_code, payload=payload, csrf_token=csrf_token, url=url)


def resume_app(opener: Any, app_url: str, csrf_token: str | None, timeout_seconds: int) -> None:
    """Ask Streamlit Cloud to resume a sleeping app."""

    url = api_url(app_url, "/api/v2/app/resume")
    http_code, payload, _csrf_token = request_json(
        opener,
        url,
        method="POST",
        csrf_token=csrf_token,
        timeout_seconds=timeout_seconds,
    )
    if http_code not in {200, 202, 204}:
        raise StreamlitKeepaliveError(
            f"Streamlit Cloud resume request failed with HTTP {http_code}: {payload}"
        )


def wait_until_running(
    app_url: str,
    *,
    timeout_seconds: int,
    poll_seconds: int,
    request_timeout_seconds: int,
) -> None:
    """Resume the app if needed and wait until Streamlit reports RUNNING."""

    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    deadline = time.monotonic() + timeout_seconds
    resume_attempted = False
    last_status: int | None = None

    while time.monotonic() < deadline:
        status_response = fetch_status(opener, app_url, request_timeout_seconds)
        last_status = status_response.app_status
        print(
            "Streamlit status: "
            f"{status_label(last_status)} "
            f"(status={last_status}, http={status_response.http_code})"
        )

        if status_response.http_code != 200:
            raise StreamlitKeepaliveError(
                f"Streamlit Cloud status request failed with HTTP {status_response.http_code}: "
                f"{status_response.payload}"
            )

        if last_status == RUNNING_STATUS:
            print("Streamlit app backend is running.")
            return

        if last_status == SHUTDOWN_STATUS and not resume_attempted:
            print("Streamlit app is sleeping. Sending resume request.")
            resume_app(opener, app_url, status_response.csrf_token, request_timeout_seconds)
            resume_attempted = True
        elif last_status not in BOOTING_STATUSES and last_status != SHUTDOWN_STATUS:
            raise StreamlitKeepaliveError(
                f"Streamlit app is not healthy: {status_label(last_status)}."
            )

        time.sleep(poll_seconds)

    raise StreamlitKeepaliveError(
        f"Timed out waiting for Streamlit app to run. Last status: {status_label(last_status)}."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the keepalive helper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("STREAMLIT_APP_URL", ""),
        help="Streamlit app URL. Defaults to STREAMLIT_APP_URL.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=540,
        help="Total time to wait for the app backend to run.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=15,
        help="Seconds to wait between Streamlit status checks.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=45,
        help="Timeout for each individual Streamlit Cloud request.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the keepalive check and return a process exit code."""

    args = parse_args()
    try:
        app_url = normalize_app_url(args.url)
        print(f"Checking Streamlit app: {app_url}")
        wait_until_running(
            app_url,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    except StreamlitKeepaliveError as error:
        print(f"Streamlit keepalive failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
