#!/usr/bin/env python3
"""
CivitAI Model Downloader - Downloads AI models from CivitAI with intelligent file handling.
Supports automatic ZIP extraction, safetensors filtering, and robust error recovery.
Changes:
- Smart Fallback: Automatically catches failures on civitai.red and transparently retries on civitai.com.
- Advanced Parsing: Accepts both standard 'versionId:fileId' strings and raw URLs from either domain.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
import re
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlencode, unquote, urlparse, parse_qs
import requests

# Constants
CIVITAI_RED_API_BASE = "https://civitai.red/api"
CIVITAI_COM_API_BASE = "https://civitai.com/api"
DEFAULT_API_BASE = CIVITAI_RED_API_BASE
ARIA2_CONNECTIONS = 8
ARIA2_SPLITS = 8
PROGRESS_INTERVAL = 10
SAFETENSORS_EXT = ".safetensors"
ZIP_EXT = ".zip"
ARIA2_EXT = ".aria2"
MIN_FILE_MB = 1  # basic sanity threshold

# Status indicators for better UX
STATUS = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "🔍",
    "download": "📥",
    "extract": "📦",
    "cleanup": "🗑️",
    "file": "📁",
}


class CivitAIDownloader:
    """Handles downloading and processing of CivitAI model files."""

    def __init__(self, token: str, output_dir: str = ".", api_base: str = DEFAULT_API_BASE):
        self.token = token or ""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_base = api_base

    # --- Header filename helpers ------------------------------------------------

    @staticmethod
    def _parse_content_disposition_filename(header_value: str) -> Optional[str]:
        """Parse Content-Disposition to extract filename."""
        if not header_value:
            return None

        m = re.search(
            r'filename\*\s*=\s*([^\'";]+)\'\'([^;]+)', header_value, flags=re.IGNORECASE
        )
        if m:
            encoded = m.group(2)
            try:
                return unquote(encoded)
            except Exception:
                pass

        m = re.search(r'filename\s*=\s*"([^"]+)"',
                      header_value, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"filename\s*=\s*([^;]+)",
                      header_value, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

        return None

    def _resolve_redirect(self, url: str) -> Tuple[str, Optional[str]]:
        """Resolve CivitAI's redirect to get direct download URL and filename."""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            r = requests.get(url, headers=headers,
                             allow_redirects=False, timeout=30)
            if r.status_code in (301, 302, 303, 307, 308):
                resolved = r.headers["Location"]
                parsed = urlparse(resolved)
                qs = parse_qs(parsed.query)
                fname = None
                if "b2ContentDisposition" in qs:
                    cd_value = unquote(qs["b2ContentDisposition"][0])
                    fname = self._parse_content_disposition_filename(cd_value)
                return resolved, fname
            elif r.status_code == 200:
                cd = r.headers.get("Content-Disposition", "")
                fname = self._parse_content_disposition_filename(cd)
                return url, fname
            else:
                return url, None
        except requests.RequestException:
            return url, None

    # --- Metadata ---------------------------------------------------------------

    def get_model_info(self, model_id: str) -> Optional[str]:
        """Fetch model metadata from CivitAI API and return the primary model file name."""
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        url = f"{self.api_base}/v1/model-versions/{model_id}"
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "files" in data and data["files"]:
                files = data["files"]
                for f in files:
                    if f.get("primary"):
                        return f.get("name")
                for f in files:
                    if f.get("type") == "Model" and f.get("metadata", {}).get("format") == "SafeTensor":
                        return f.get("name")
                for f in files:
                    if f.get("type") == "Model":
                        return f.get("name")
                return files[0].get("name")
            return None
        except requests.RequestException:
            return None

    # --- File utilities ---------------------------------------------------------

    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate file existence and check for incomplete downloads."""
        if not file_path.exists():
            return False, "File does not exist"
        if file_path.with_suffix(file_path.suffix + ARIA2_EXT).exists():
            return False, "Incomplete download detected (aria2 control file exists)"
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb < MIN_FILE_MB:
            return False, f"File suspiciously small ({file_size_mb:.2f}MB)"
        return True, f"File valid ({file_size_mb:.1f}MB)"

    def cleanup_incomplete_download(self, file_path: Path) -> None:
        """Remove incomplete download artifacts."""
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        aria2_file = file_path.with_suffix(file_path.suffix + ARIA2_EXT)
        if aria2_file.exists():
            aria2_file.unlink(missing_ok=True)

    def extract_safetensors_from_zip(self, zip_path: Path) -> Tuple[bool, str, Optional[Path]]:
        """Extract and keep only safetensors files from ZIP archive."""
        print(f"{STATUS['extract']} Extracting: {zip_path.name}")
        temp_dir = self.output_dir / f"temp_extract_{zip_path.stem}"
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                safetensors_in_zip = [n for n in zip_ref.namelist(
                ) if n.lower().endswith(SAFETENSORS_EXT)]
                if not safetensors_in_zip:
                    return True, "No safetensors files in archive - keeping original ZIP", None
                temp_dir.mkdir(exist_ok=True)
                for file_name in safetensors_in_zip:
                    zip_ref.extract(file_name, temp_dir)
                moved_count = 0
                last_moved: Optional[Path] = None
                for extracted_file in temp_dir.rglob(f"*{SAFETENSORS_EXT}"):
                    dest_file = self._get_unique_filename(
                        self.output_dir / extracted_file.name)
                    shutil.move(str(extracted_file), str(dest_file))
                    print(f"{STATUS['file']} Extracted: {dest_file.name}")
                    moved_count += 1
                    last_moved = dest_file
                shutil.rmtree(temp_dir, ignore_errors=True)
                zip_path.unlink(missing_ok=True)
                return True, f"Extracted {moved_count} safetensors file(s)", last_moved
        except zipfile.BadZipFile:
            return False, "Corrupted or invalid ZIP file", None
        except Exception as e:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return False, f"Extraction error: {e}", None

    def _get_unique_filename(self, file_path: Path) -> Path:
        """Generate unique filename if conflict exists."""
        if not file_path.exists():
            return file_path
        counter = 1
        while True:
            new_path = file_path.parent / \
                f"{file_path.stem}_{counter}{file_path.suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def process_downloaded_file(self, file_path: Path) -> Tuple[bool, str, Optional[Path]]:
        """Process downloaded file based on its type."""
        if not file_path.exists():
            return False, "Downloaded file not found", None
        suffix = file_path.suffix.lower()
        if suffix == ZIP_EXT:
            return self.extract_safetensors_from_zip(file_path)
        elif suffix == SAFETENSORS_EXT:
            return True, "File ready to use", file_path
        else:
            return True, "File downloaded successfully", file_path

    # --- Download core ----------------------------------------------------------

    def _download_with_url(self, download_url: str, prefer_filename: Optional[str], force: bool = False) -> Tuple[bool, Optional[Path]]:
        """Download file using aria2c with the given URL."""
        resolved_url, redirect_filename = self._resolve_redirect(download_url)
        filename = redirect_filename or prefer_filename or "download.bin"
        file_path = self.output_dir / filename

        is_resuming = False

        if not force and file_path.exists():
            is_valid, message = self.validate_file(file_path)
            if is_valid:
                print(
                    f"{STATUS['success']} File already exists and is valid: {file_path.name}")
                return True, file_path
            elif "aria2 control file exists" in message:
                is_resuming = True
                print(
                    f"{STATUS['info']} Partial download detected. Resuming {file_path.name}...")
            else:
                self.cleanup_incomplete_download(file_path)

        if not force and not is_resuming:
            file_path = self._get_unique_filename(file_path)

        cmd = [
            "aria2c",
            f"--max-connection-per-server={ARIA2_CONNECTIONS}",
            f"--split={ARIA2_SPLITS}",
            "--min-split-size=4M",
            "--continue=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--max-tries=4",
            "--retry-wait=2",
            "--timeout=30",
            "--connect-timeout=10",
            f"--summary-interval={PROGRESS_INTERVAL}",
            "--console-log-level=warn",
            f"--dir={self.output_dir}",
            f"--out={file_path.name}",
        ]

        if self.token:
            cmd.append(f"--header=Authorization: Bearer {self.token}")

        cmd.append(resolved_url)

        try:
            subprocess.run(cmd, check=True)

            actual = file_path if file_path.exists() else None
            if not actual:
                all_files = list(self.output_dir.glob("*"))
                recent_files = [f for f in all_files if f.is_file() and (
                    time.time() - f.stat().st_mtime) < 60]
                if recent_files:
                    actual = max(recent_files, key=lambda f: f.stat().st_mtime)

            if not actual:
                return False, None

            is_valid, message = self.validate_file(actual)
            if not is_valid:
                print(f"{STATUS['error']} Validation failed: {message}")
                return False, None

            return True, actual
        except subprocess.CalledProcessError:
            return False, None

    def download_with_aria2(self, model_id: str, file_id: str, prefer_filename: Optional[str], force: bool = False) -> Tuple[bool, Optional[Path]]:
        """Try downloading the primary file, then try Diffusers ZIP as a backup."""
        params = {"type": "Model", "fileId": file_id}
        primary_url = f"{self.api_base}/download/models/{model_id}?{urlencode(params)}"

        target_name = prefer_filename or self.get_model_info(model_id)

        ok, path = self._download_with_url(primary_url, target_name, force)
        if ok and path:
            ok2, msg, final_path = self.process_downloaded_file(path)
            if ok2:
                return True, final_path or path

        # --- Backup Attempt: Diffusers ZIP ---
        params = {"type": "Model", "format": "Diffusers", "fileId": file_id}
        zip_url = f"{self.api_base}/download/models/{model_id}?{urlencode(params)}"
        header_name = f"{Path(target_name).stem}_diffusers.zip" if target_name else None

        ok, path = self._download_with_url(zip_url, header_name, force)
        if ok and path:
            ok2, msg, final_path = self.process_downloaded_file(path)
            if ok2:
                return True, final_path or path

        return False, None


def get_token(args_token: Optional[str]) -> str:
    """Retrieve CivitAI token from environment or arguments."""
    token = os.getenv("CIVITAI_TOKEN") or os.getenv("civitai_token")
    if token:
        return token
    if args_token:
        return args_token
    print(f"{STATUS['error']} No CivitAI token provided.")
    sys.exit(1)


def parse_model_arg(raw: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse the -m argument into (model_id, file_id, detected_api_base).
    Supports raw versionId:fileId strings and full links from either domain.
    """
    raw = raw.strip()

    # Case 1: Raw Browser URLs or Download API Links
    if raw.startswith("http://") or raw.startswith("https://"):
        try:
            parsed = urlparse(raw)
            detected_base = CIVITAI_COM_API_BASE if "civitai.com" in parsed.netloc else CIVITAI_RED_API_BASE
            qs = parse_qs(parsed.query)
            file_id = qs.get("fileId", [None])[0]
            model_id = None

            path_parts = parsed.path.strip("/").split("/")
            if "models" in path_parts:
                idx = path_parts.index("models")
                if idx + 1 < len(path_parts) and path_parts[idx + 1].isdigit():
                    model_id = path_parts[idx + 1]

            if not model_id and "modelVersionId" in qs:
                model_id = qs["modelVersionId"][0]

            if model_id and file_id:
                return model_id, file_id, detected_base

            print(
                f"{STATUS['error']} URL must contain both model/version ID and fileId query parameter.")
            sys.exit(1)
        except Exception as e:
            print(f"{STATUS['error']} Failed parsing URL format: {e}")
            sys.exit(1)

    # Case 2: Standard expected string layout -> versionId:fileId
    if ":" in raw:
        model_id, file_id = raw.split(":", 1)
        if model_id.strip().isdigit() and file_id.strip().isdigit():
            return model_id.strip(), file_id.strip(), None

    print(f"{STATUS['error']} Invalid argument target layout: '{raw}'")
    print("  Expected either: 'versionId:fileId' or a full CivitAI API link containing '?fileId=...'")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download AI models from CivitAI with automatic fallback handling.")
    parser.add_argument("-m", "--model-id", required=True,
                        help="versionId:fileId or a full CivitAI URL.")
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("--token", help="CivitAI API token")
    parser.add_argument("--filename", help="Override filename")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download")
    parser.add_argument("--base-url", default=DEFAULT_API_BASE,
                        help="Override initial base URL setup")
    args = parser.parse_args()

    try:
        token = get_token(args.token)
        model_id, file_id, detected_base = parse_model_arg(args.model_id)

        # Decide starting domain based on URL format or configuration flags
        base_url = args.base_url
        if args.base_url == DEFAULT_API_BASE and detected_base:
            base_url = detected_base

        print(
            f"{STATUS['info']} Target Details -> Version: {model_id} | File ID: {file_id}")
        print(f"{STATUS['info']} Primary Request Endpoint Base: {base_url}")

        downloader = CivitAIDownloader(token, args.output, api_base=base_url)

        # First Attempt
        ok, final_path = downloader.download_with_aria2(
            model_id, file_id, args.filename, force=args.force)

        # Automatic Switch to civitai.com fallback if the first target was civitai.red and failed
        if not ok and base_url == CIVITAI_RED_API_BASE:
            print(f"\n{STATUS['warning']} Primary download mirror failed.")
            print(
                f"{STATUS['info']} Switching API Base to Main Site (civitai.com) and retrying download...")
            downloader.api_base = CIVITAI_COM_API_BASE
            ok, final_path = downloader.download_with_aria2(
                model_id, file_id, args.filename, force=args.force)

        if ok:
            if final_path and final_path.exists():
                print(f"{STATUS['success']} Model ready at: {final_path}")
            else:
                print(f"{STATUS['success']} Download completed successfully")
        else:
            print(
                f"{STATUS['error']} All API endpoints exhausted. Download aborted.")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n{STATUS['warning']} Process terminated by user.")
        sys.exit(130)
    except Exception as e:
        print(f"{STATUS['error']} Crash exception caught: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
