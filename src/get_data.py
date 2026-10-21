"""Download the starter photos into data/starter/.

This file is complete.

    python src/get_data.py --link "<the link on Cyber Campus>"

The photos are for the students of this class only, so the link is on Cyber
Campus and not in this repository. The first run keeps it in
data/starter_link.txt (data/ is not pushed to GitHub); later runs find it
there.

data/starter/train/<city>/   training photos (Track A may use only these)
data/starter/val/<city>/     validation photos, by different photographers
data/starter/ATTRIBUTION.csv the author, licence and source page of every photo
data/sample_target/<city>/   ten photos per city taken the way the final test photos were

The photos are from Wikimedia Commons, OSV-5M (Mapillary) and YouTube
walking and driving videos; ATTRIBUTION.csv gives the author, licence and
source of each. They are for the students of this class only. The shorter
side of every photo is 256 pixels.
"""

import hashlib
import html
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Filled in when the data is published (see planning/assignment2-city-classifier-design.md).
SHA256 = "b9e8f5748604b972f5c4a234855d1b2fcfaa77daebc903320acbc7c0b9ca363e"
LINK_FILE = ROOT / "data" / "starter_link.txt"
EXPECTED = {"train": 44257, "val": 12077}


def drive_file_id(link):
    m = re.search(r"/d/([\w-]{20,})", link) or re.search(r"[?&]id=([\w-]{20,})", link)
    if not m:
        sys.exit(f"This does not look like a Google Drive link to a file: {link}")
    return m.group(1)


def open_drive(file_id):
    """Open the file for reading. For a large file Google Drive first sends a
    page saying it cannot scan it for viruses, with a form to download anyway."""
    r = urllib.request.urlopen("https://drive.usercontent.google.com/download?"
                               + urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"}))
    if r.headers.get_content_type() == "text/html":
        page = r.read().decode("utf-8", "replace")
        form = re.search(r'<form[^>]*action="([^"]+)"', page)
        if form:
            fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page))
            r = urllib.request.urlopen(html.unescape(form.group(1)) + "?" + urllib.parse.urlencode(fields))
    if r.headers.get_content_type() == "text/html":
        sys.exit("Google Drive did not send the file. If many people downloaded it today, Drive may "
                 "refuse for a while: try again in a few hours, or ask in class.")
    return r


def download(file_id, dest):
    with open_drive(file_id) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} of {total / 1e6:.0f} MB", end="", flush=True)
    print()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", help="the link to the starter photos on Cyber Campus")
    ap.add_argument("--into", default=str(ROOT / "data"),
                    help="where starter/ and sample_target/ go (on Colab, the fast local disk)")
    args = ap.parse_args()
    data = Path(args.into)
    counts = {part: len(list((data / "starter" / part).glob("*/*.jpg"))) for part in EXPECTED}
    if counts == EXPECTED:
        print(f"data/starter is already complete: {counts}")
        return
    if args.link:
        LINK_FILE.parent.mkdir(parents=True, exist_ok=True)
        LINK_FILE.write_text(args.link.strip() + "\n")
    elif not LINK_FILE.exists():
        sys.exit('Run again with --link "<the link>": the link to the starter photos is on Cyber Campus.')
    file_id = drive_file_id(LINK_FILE.read_text().strip())
    data.mkdir(parents=True, exist_ok=True)
    zip_path = data / "starter.zip"
    print("downloading the starter photos from Google Drive")
    download(file_id, zip_path)
    if SHA256:
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if digest != SHA256:
            sys.exit(f"The download is damaged (sha256 {digest[:12]}...). Run this again.")
    for name in ("starter", "sample_target"):
        if (data / name).is_dir() and not (data / name).is_symlink():
            shutil.rmtree(data / name)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data)
    zip_path.unlink()
    counts = {part: len(list((data / "starter" / part).glob("*/*.jpg"))) for part in EXPECTED}
    if counts != EXPECTED:
        sys.exit(f"Expected {EXPECTED} photos, found {counts}.")
    print(f"data/starter: {counts['train']} training and {counts['val']} validation photos")


if __name__ == "__main__":
    main()
