from __future__ import annotations

# Standard library imports
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

# Third-party imports
import requests
from dotenv import load_dotenv


# Base ACN API session endpoint
BASE_URL = "https://ev.caltech.edu/api/v1/sessions"
API_ROOT = "https://ev.caltech.edu/api/v1/"


def get_token() -> str:
    """
    Load the ACN API token from the local .env file.
    This token is kept local and is NOT pushed to GitHub.
    """
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
    """
    Fetch EV charging sessions from the ACN API.

    Parameters
    ----------
    site : str
        Site ID, such as 'caltech'.
    where : str | None
        Optional ACN API filter string.
    sort : str | None
        Optional sort field, e.g. 'connectionTime' or '-connectionTime'.
    max_pages : int | None
        Optional limit for debugging. If None, fetch all pages.

    Returns
    -------
    list[dict[str, Any]]
        List of charging session records from the API.
    """
    token = get_token()

    # Start at the site-specific endpoint
    url = f"{BASE_URL}/{site}"
    params = {}

    # Add optional filters
    if where:
        params["where"] = where
    if sort:
        params["sort"] = sort

    sessions: list[dict[str, Any]] = []
    page = 1

    # Loop through all API pages until no "next" page exists
    while url:
        print(f"\nFetching page {page} ...")
        print(f"URL: {url}")
        if params:
            print(f"Params: {params}")

        response = requests.get(
            url,
            params=params if "?" not in url else None,
            auth=(token, ""),   # ACN uses token as username, blank password
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()

        # Extract the session list from the current page
        page_items = data.get("_items", [])
        sessions.extend(page_items)

        print(f"Retrieved {len(page_items)} sessions on this page.")
        print(f"Total sessions so far: {len(sessions)}")

        # Optional stop for debugging or testing small fetches
        if max_pages is not None and page >= max_pages:
            print("\nReached max_pages limit. Stopping early.")
            break

        # Follow ACN pagination link if it exists
        next_link = data.get("_links", {}).get("next", {})
        href = next_link.get("href")

        if href:
            if href.startswith("http"):
                url = href
            else:
                # ACN next links are relative to API root
                url = urljoin(API_ROOT, href)
            params = None
            page += 1
        else:
            url = None
            print("\nNo more pages. Download complete.")

    return sessions


def save_sessions_to_json(sessions: list[dict[str, Any]], output_path: str) -> None:
    """
    Save raw API session records to a local JSON file.
    This is our raw dataset layer.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)

    print(f"\nSaved {len(sessions)} sessions to {path}")


def main() -> None:
    """
    Main data acquisition step:
    fetch the Caltech ACN sessions for the selected date range,
    then save them into the raw data folder.
    """
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