#!/usr/bin/env python3
"""Automate Label Studio project setup + pre-annotation import via the REST API.

On a fresh Label Studio instance this will:
  1. sign up the first user (or log in if it already exists),
  2. fetch the API token,
  3. create-or-reuse a project with the given labeling config,
  4. import one or more pre-annotation JSON files (tasks + predictions).

Run against a Label Studio server started with LOCAL_FILES_SERVING enabled and the
document root set so the `/data/local-files/?d=...` image URLs resolve.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import requests


def get_csrf(session, url):
    r = session.get(url)
    r.raise_for_status()
    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', r.text)
    token = m.group(1) if m else session.cookies.get("csrftoken")
    return token


def ensure_auth(base, email, password):
    s = requests.Session()
    # Try signup first (fresh instance); fall back to login.
    signup_url = f"{base}/user/signup/"
    csrf = get_csrf(s, signup_url)
    r = s.post(
        signup_url,
        data={"csrfmiddlewaretoken": csrf, "email": email, "password": password},
        headers={"Referer": signup_url},
        allow_redirects=False,
    )
    if r.status_code in (301, 302):
        print(f"  signed up new user: {email}")
    else:
        login_url = f"{base}/user/login/"
        csrf = get_csrf(s, login_url)
        r = s.post(
            login_url,
            data={"csrfmiddlewaretoken": csrf, "email": email, "password": password},
            headers={"Referer": login_url},
            allow_redirects=False,
        )
        if r.status_code not in (301, 302):
            print(f"  ERROR: could not sign up or log in ({r.status_code}). "
                  f"The email may exist with a different password.", file=sys.stderr)
            sys.exit(1)
        print(f"  logged in existing user: {email}")

    tok = s.get(f"{base}/api/current-user/token")
    tok.raise_for_status()
    token = tok.json().get("token")
    if not token:
        print("  ERROR: no API token returned.", file=sys.stderr)
        sys.exit(1)
    return token


def api(base, token):
    h = {"Authorization": f"Token {token}"}
    return h


def get_or_create_project(base, headers, title, label_config):
    r = requests.get(f"{base}/api/projects", headers=headers, params={"page_size": 1000})
    r.raise_for_status()
    for p in r.json().get("results", []):
        if p["title"] == title:
            print(f"  project '{title}' already exists (id={p['id']})")
            return p["id"]
    r = requests.post(
        f"{base}/api/projects",
        headers=headers,
        json={"title": title, "label_config": label_config},
    )
    r.raise_for_status()
    pid = r.json()["id"]
    print(f"  created project '{title}' (id={pid})")
    return pid


def import_tasks(base, headers, pid, json_path):
    tasks = json.loads(Path(json_path).read_text())
    r = requests.post(
        f"{base}/api/projects/{pid}/import",
        headers=headers,
        json=tasks,
    )
    r.raise_for_status()
    j = r.json()
    print(f"  imported {json_path}: {j.get('task_count')} tasks, "
          f"{j.get('prediction_count')} predictions, {j.get('annotation_count')} annotations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--config", required=True, help="Path to labeling config XML.")
    ap.add_argument("--project", required=True, help="Project title.")
    ap.add_argument("--import", dest="imports", nargs="*", default=[],
                    help="Pre-annotation JSON files to import into the project.")
    ap.add_argument("--print-token", action="store_true")
    args = ap.parse_args()

    label_config = Path(args.config).read_text()
    print("Authenticating...")
    token = ensure_auth(args.base, args.email, args.password)
    if args.print_token:
        print(f"  API token: {token}")
    headers = api(args.base, token)

    print("Setting up project...")
    pid = get_or_create_project(args.base, headers, args.project, label_config)
    for jp in args.imports:
        import_tasks(args.base, headers, pid, jp)
    print("Done.")


if __name__ == "__main__":
    main()
