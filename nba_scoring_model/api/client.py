from collections import deque
import threading
import time
from typing import Any, Dict, Iterator, Mapping, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RateLimiter:
    def __init__(self, calls: int = 60, period_seconds: float = 60.0):
        if calls <= 0 or period_seconds <= 0:
            raise ValueError("calls and period_seconds must be positive")
        self.calls = calls
        self.period_seconds = period_seconds
        self.timestamps = deque()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            while self.timestamps and now - self.timestamps[0] >= self.period_seconds:
                self.timestamps.popleft()

            if len(self.timestamps) >= self.calls:
                sleep_for = self.period_seconds - (now - self.timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= self.period_seconds:
                    self.timestamps.popleft()

            self.timestamps.append(time.monotonic())


class JSONAPIClient:
    def __init__(
        self,
        calls_per_minute: int = 60,
        timeout_seconds: float = 15.0,
        headers: Optional[Mapping[str, str]] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.rate_limiter = RateLimiter(calls_per_minute, 60.0)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                **dict(headers or {}),
            }
        )

        retry = Retry(
            total=2,
            connect=2,
            read=2,
            status=2,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.rate_limiter.wait()
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object from {url}")
        return payload

    def iter_paginated(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        items_key: str = "data",
        next_key: str = "next",
        page_param: str = "page",
        page: int = 1,
        max_pages: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        request_params = dict(params or {})
        pages_seen = 0
        next_value: Any = page

        while next_value is not None:
            if max_pages is not None and pages_seen >= max_pages:
                break

            request_params[page_param] = next_value
            payload = self.get_json(url, request_params)
            items = payload.get(items_key, [])
            if not isinstance(items, list):
                raise ValueError(f"Expected '{items_key}' to be a list")

            for item in items:
                if isinstance(item, dict):
                    yield item

            pages_seen += 1
            next_value = payload.get(next_key)

            if next_value is None:
                break
