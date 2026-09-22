from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import requests


logger = logging.getLogger(__name__)


class KisApiError(RuntimeError):
    """Raised when KIS Open API authentication or data retrieval fails."""


_RATE_LIMIT_CODE = "EGW00201"


@dataclass
class KisOpenApiClient:
    """Minimal read-only KIS Open API REST client for research.

    v0.4.3 adds three safeguards that are important for batch research:
    1) token reuse across processes via a local cache,
    2) throttling *after* authentication so the first GET is not sent
       immediately after a token request,
    3) automatic retry/backoff for EGW00201 (per-second call limit).
    """

    app_key: str
    app_secret: str
    base_url: str = "https://openapi.koreainvestment.com:9443"
    pause_seconds: float = 1.0
    timeout_seconds: float = 20.0
    rate_limit_wait_seconds: float = 61.0
    rate_limit_max_retries: int = 3
    retry_jitter_seconds: float = 0.5
    token_cache_path: str | None = None
    session: requests.Session = field(default_factory=requests.Session)

    _access_token: str | None = field(default=None, init=False, repr=False)
    _token_expiry_epoch: float = field(default=0.0, init=False, repr=False)
    _last_request_monotonic: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "KisOpenApiClient":
        app_key = os.environ.get("KIS_APP_KEY", "").strip()
        app_secret = os.environ.get("KIS_APP_SECRET", "").strip()
        if not app_key or not app_secret:
            raise KisApiError(
                "KIS credentials are missing. Set KIS_APP_KEY and KIS_APP_SECRET."
            )
        base_url = os.environ.get(
            "KIS_BASE_URL", "https://openapi.koreainvestment.com:9443"
        ).rstrip("/")
        pause = float(os.environ.get("KIS_PAUSE_SECONDS", kwargs.pop("pause_seconds", 1.0)))
        wait = float(
            os.environ.get(
                "KIS_RATE_LIMIT_WAIT_SECONDS",
                kwargs.pop("rate_limit_wait_seconds", 61.0),
            )
        )
        retries = int(
            os.environ.get(
                "KIS_RATE_LIMIT_MAX_RETRIES",
                kwargs.pop("rate_limit_max_retries", 3),
            )
        )
        token_cache_path = os.environ.get(
            "KIS_TOKEN_CACHE",
            kwargs.pop("token_cache_path", None),
        )
        return cls(
            app_key=app_key,
            app_secret=app_secret,
            base_url=base_url,
            pause_seconds=pause,
            rate_limit_wait_seconds=wait,
            rate_limit_max_retries=retries,
            token_cache_path=token_cache_path,
            **kwargs,
        )

    def _cache_file(self) -> Path:
        if self.token_cache_path:
            return Path(self.token_cache_path).expanduser()
        key_id = hashlib.sha256(
            f"{self.base_url}|{self.app_key}".encode("utf-8")
        ).hexdigest()[:16]
        return Path.home() / ".cache" / "korea_sd" / f"kis_token_{key_id}.json"

    def _load_cached_token(self) -> bool:
        path = self._cache_file()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        token = str(data.get("access_token", "")).strip()
        try:
            expiry = float(data.get("expiry_epoch", 0))
        except (TypeError, ValueError):
            return False
        # Refresh at least 5 minutes early.
        if not token or time.time() >= expiry - 300:
            return False
        self._access_token = token
        self._token_expiry_epoch = expiry
        return True

    def _save_cached_token(self, token: str, expiry_epoch: float) -> None:
        path = self._cache_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {"access_token": token, "expiry_epoch": expiry_epoch},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Could not write KIS token cache %s: %s", path, exc)

    def _throttle(self) -> None:
        if self.pause_seconds <= 0 or self._last_request_monotonic <= 0:
            return
        elapsed = time.monotonic() - self._last_request_monotonic
        wait = self.pause_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    def _ensure_token(self) -> str:
        if self._access_token and time.time() < self._token_expiry_epoch - 300:
            return self._access_token
        if self._load_cached_token():
            return self._access_token or ""

        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise KisApiError(f"KIS token request failed: {exc}") from exc
        finally:
            # Token issuance is also a network/API operation. Record it so the
            # first market-data GET is not sent immediately afterwards.
            self._last_request_monotonic = time.monotonic()

        if response.status_code != 200:
            raise KisApiError(
                f"KIS token request failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise KisApiError(
                f"KIS token request returned non-JSON response: {response.text[:500]}"
            ) from exc

        token = str(body.get("access_token", "")).strip()
        if not token:
            raise KisApiError(f"KIS token response did not contain access_token: {body}")

        try:
            expires_in = int(body.get("expires_in", 23 * 60 * 60))
        except (TypeError, ValueError):
            expires_in = 23 * 60 * 60

        expiry_epoch = time.time() + max(300, expires_in)
        self._access_token = token
        self._token_expiry_epoch = expiry_epoch
        self._save_cached_token(token, expiry_epoch)
        return token

    @staticmethod
    def _response_body(response: requests.Response) -> dict[str, Any] | None:
        try:
            body = response.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    @classmethod
    def _is_rate_limit_response(
        cls, response: requests.Response, body: dict[str, Any] | None
    ) -> bool:
        if body:
            code = str(body.get("msg_cd") or body.get("message") or "")
            if _RATE_LIMIT_CODE in code:
                return True
            if _RATE_LIMIT_CODE in str(body.get("msg1") or ""):
                return True
        return _RATE_LIMIT_CODE in (response.text or "")

    def _rate_limit_sleep(self, attempt: int) -> None:
        base = max(0.0, self.rate_limit_wait_seconds)
        jitter = random.uniform(0.0, max(0.0, self.retry_jitter_seconds))
        wait = base + jitter
        logger.warning(
            "KIS rate limit %s; waiting %.1fs before retry (%d/%d)",
            _RATE_LIMIT_CODE,
            wait,
            attempt,
            self.rate_limit_max_retries,
        )
        time.sleep(wait)

    def get(
        self,
        api_path: str,
        tr_id: str,
        params: Mapping[str, Any],
        *,
        tr_cont: str = "",
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        """Call a read-only KIS endpoint and return ``(JSON body, headers)``.

        EGW00201 is retried automatically. Other API errors remain hard
        failures so research downloads never silently create partial files.
        """

        token = self._ensure_token()
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        if tr_cont:
            headers["tr_cont"] = tr_cont

        url = f"{self.base_url}{api_path}"
        attempts = max(1, self.rate_limit_max_retries)
        for attempt in range(1, attempts + 1):
            # Important ordering: authenticate first, then throttle the GET.
            self._throttle()
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    params={k: "" if v is None else str(v) for k, v in params.items()},
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                raise KisApiError(f"KIS GET failed for {api_path}: {exc}") from exc
            finally:
                self._last_request_monotonic = time.monotonic()

            body = self._response_body(response)
            if self._is_rate_limit_response(response, body):
                if attempt < attempts:
                    self._rate_limit_sleep(attempt)
                    continue
                raise KisApiError(
                    f"KIS GET {api_path} hit {_RATE_LIMIT_CODE} after {attempts} attempts: "
                    f"{response.text[:500]}"
                )

            if response.status_code != 200:
                raise KisApiError(
                    f"KIS GET {api_path} failed with HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            if body is None:
                raise KisApiError(
                    f"KIS GET {api_path} returned non-JSON response: {response.text[:500]}"
                )

            rt_cd = body.get("rt_cd")
            if rt_cd not in (None, "0", 0):
                raise KisApiError(
                    f"KIS GET {api_path} failed: rt_cd={rt_cd}, "
                    f"msg_cd={body.get('msg_cd')}, msg1={body.get('msg1')}"
                )

            return body, response.headers

        raise KisApiError(f"KIS GET {api_path} failed unexpectedly")
