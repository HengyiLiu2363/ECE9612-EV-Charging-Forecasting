from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv


BASE_URL = "https://ev.caltech.edu/api/v1/sessions"
API_ROOT = "https://ev.caltech.edu/api/v1/"


def get_token() -> str:
    load_dotenv()
    token = os.getenv("ACN_API_TOKEN")
    if not token:
        raise ValueError("ACN_API_TOKEN not found in .env")
    return token


def fetch_sessions(
    site: str = "caltech",
    where: str | None = None,
    sort: str | None = None,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    token = get_token()

    url = f"{BASE_URL}/{site}"
    params = {}

    if where:
        params["where"] = where
    if sort:
        params["sort"] = sort

    sessions: list[dict[str, Any]] = []
    page = 1

    while url:
        print(f"\nFetching page {page} ...")
        print(f"URL: {url}")
        if params:
            print(f"Params: {params}")

        response = requests.get(
            url,
            params=params if "?" not in url else None,
            auth=(token, ""),
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        page_items = data.get("_items", [])
        sessions.extend(page_items)

        print(f"Retrieved {len(page_items)} sessions on this page.")
        print(f"Total sessions so far: {len(sessions)}")

        if max_pages is not None and page >= max_pages:
            print("\nReached max_pages limit. Stopping early.")
            break

        next_link = data.get("_links", {}).get("next", {})
        href = next_link.get("href")

        if href:
            if href.startswith("http"):
                url = href
            else:
                url = urljoin(API_ROOT, href)
            params = None
            page += 1
        else:
            url = None
            print("\nNo more pages. Download complete.")

    return sessions


def save_sessions_to_json(sessions: list[dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)

    print(f"\nSaved {len(sessions)} sessions to {path}")


def main() -> None:
    where_clause = (
        'connectionTime >= "Tue, 1 Jan 2019 00:00:00 GMT" and '
        'connectionTime <= "Tue, 14 Sep 2021 23:59:59 GMT"'
    )

    sessions = fetch_sessions(
        site="caltech",
        where=where_clause,
        sort="connectionTime",
        max_pages=None,
    )

    save_sessions_to_json(sessions, "data/raw/acn_caltech_sessions_2019_2021.json")


if __name__ == "__main__":
    main()