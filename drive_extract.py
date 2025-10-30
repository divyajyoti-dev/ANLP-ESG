#!/usr/bin/env python3
"""
Sync PDFs from Google Drive folders (by folder ID) into local directories.

Auth:
- Put your OAuth creds in ./auth (credentials.json + token.json) or a service
  account JSON as ./auth/service_account.json (auto-detected).
Usage:
  python drive_sync.py --dest data/raw --folder "GOOGLE=1AbC...xyz" "APPLE=1Def...uvw"
  # or: python drive_sync.py --dest data/raw --company GOOGLE --folder-id 1AbC...xyz
"""

import os
import io
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.auth.exceptions

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
AUTH_DIR = Path("auth")
TOKEN_PATH = AUTH_DIR / "token.json"
CRED_PATH = AUTH_DIR / "credentials.json"
SA_PATH = AUTH_DIR / "service_account.json"


def _service_via_service_account():
    creds = service_account.Credentials.from_service_account_file(
        SA_PATH, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _service_via_oauth():
    creds = None
    if TOKEN_PATH.exists():
        try:
            from google.oauth2.credentials import Credentials as UserCreds
            creds = UserCreds.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CRED_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_drive_service():
    """
    Prefer service account if available; otherwise OAuth user creds.
    """
    if SA_PATH.exists():
        return _service_via_service_account()
    if CRED_PATH.exists():
        return _service_via_oauth()
    raise FileNotFoundError(
        "No auth found. Provide auth/service_account.json or auth/credentials.json"
    )


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _file_exists_with_md5(local_path: Path, md5: Optional[str]) -> bool:
    """
    Skip download if a file exists AND we have an md5 to compare.
    If Drive doesn’t provide md5 (rare for Google Docs), we fall back to name-only.
    """
    if not local_path.exists():
        return False
    if not md5:
        return True
    # Store md5 next to file on first download and compare later
    md5_path = local_path.with_suffix(local_path.suffix + ".md5")
    return md5_path.exists() and md5_path.read_text().strip() == md5


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5))
def list_pdfs(service, folder_id: str) -> List[dict]:
    """
    List PDF files in a Drive folder (not trashed).
    """
    files = []
    page_token = None
    q = (
        f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    )
    fields = "nextPageToken, files(id, name, mimeType, size, md5Checksum, modifiedTime)"
    while True:
        resp = service.files().list(
            q=q,
            fields=fields,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5))
def download_file(service, file_id: str, dest_path: Path, md5: Optional[str]):
    """
    Download a Drive file by ID to dest_path. Writes an .md5 sidecar when available.
    """
    ensure_dir(dest_path.parent)
    if _file_exists_with_md5(dest_path, md5):
        return "skipped"

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    if md5:
        dest_path.with_suffix(dest_path.suffix + ".md5").write_text(md5)
    return "downloaded"


def sync_folder(folder_name: str, folder_id: str, dest_root: Path) -> Dict[str, int]:
    """
    Sync all PDFs in Drive folder -> local dest_root/folder_name
    Returns counts: downloaded, skipped, errors.
    """
    service = get_drive_service()
    out_dir = dest_root / folder_name
    ensure_dir(out_dir)

    counts = {"downloaded": 0, "skipped": 0, "errors": 0}
    try:
        files = list_pdfs(service, folder_id)
    except HttpError as e:
        print(f"[{folder_name}] List error: {e}")
        counts["errors"] += 1
        return counts

    for f in tqdm(files, desc=f"Sync {folder_name}", unit="file"):
        name = f["name"]
        md5 = f.get("md5Checksum")
        file_id = f["id"]
        dest = out_dir / name
        try:
            status = download_file(service, file_id, dest, md5)
            counts[status] += 1
        except HttpError as e:
            print(f"[{folder_name}] Download error for {name}: {e}")
            counts["errors"] += 1
    return counts


def parse_kv_pairs(pairs: List[str]) -> Dict[str, str]:
    """
    Parse ["GOOGLE=folderId", "APPLE=folderId2"] into dict.
    """
    out = {}
    for kv in pairs:
        if "=" not in kv:
            raise ValueError(f"Invalid --folder item: {kv}. Use NAME=FOLDER_ID")
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw", help="Root download dir")
    ap.add_argument("--company", help="Single company name (dest subdir)")
    ap.add_argument("--folder-id", help="Single folder id (with --company)")
    ap.add_argument(
        "--folder",
        nargs="*",
        help='Multiple NAME=FOLDER_ID mappings, e.g. --folder "GOOGLE=1Abc" "APPLE=2Def"',
    )
    args = ap.parse_args()

    dest_root = Path(args.dest)

    folder_map: Dict[str, str] = {}
    if args.company and args.folder_id:
        folder_map[args.company] = args.folder_id
    if args.folder:
        folder_map.update(parse_kv_pairs(args.folder))

    if not folder_map:
        raise SystemExit(
            "Provide --company and --folder-id, or one/more --folder NAME=ID pairs."
        )

    results = {}
    for name, fid in folder_map.items():
        counts = sync_folder(name, fid, dest_root)
        results[name] = counts

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
