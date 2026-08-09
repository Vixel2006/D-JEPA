import argparse
import os
import sys
import time
import urllib.request
import zipfile

BASE_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k"
FILES = [
    ("Flickr8k_Dataset.zip", 1115419746),
    ("Flickr8k_text.zip", 2340801),
]
EXPECTED_IMAGES = 8091
EXPECTED_CAPTIONS = 5 * EXPECTED_IMAGES
SPLITS = [
    ("Flickr_8k.trainImages.txt", 6000),
    ("Flickr_8k.devImages.txt", 1000),
    ("Flickr_8k.testImages.txt", 1000),
]


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def already_installed(data_dir):
    img_dir = os.path.join(data_dir, "Flicker8k_Dataset")
    token = os.path.join(data_dir, "Flickr8k.token.txt")
    return os.path.isdir(img_dir) and os.path.isfile(token)


def download(url, dest, expected_size):
    if os.path.exists(dest) and os.path.getsize(dest) == expected_size:
        print(f"  already downloaded: {os.path.basename(dest)}")
        return
    resume = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {"Range": f"bytes={resume}-"} if resume else {}
    print(f"  downloading {os.path.basename(dest)} ...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        chunk_total = int(resp.headers.get("Content-Length") or 0)
        is_resume = resp.status == 206
        done = resume if is_resume else 0
        total = done + chunk_total
        start = time.monotonic()
        with open(dest, "ab" if is_resume else "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                elapsed = max(time.monotonic() - start, 1e-6)
                if total:
                    pct = 100.0 * done / total
                    sys.stdout.write(
                        f"\r    {done / 1e6:7.1f} / {total / 1e6:7.1f} MB "
                        f"({pct:5.1f}%)  {done / elapsed / 1e6:5.1f} MB/s"
                    )
                else:
                    sys.stdout.write(
                        f"\r    {done / 1e6:7.1f} MB  {done / elapsed / 1e6:5.1f} MB/s"
                    )
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    size = os.path.getsize(dest)
    if size != expected_size:
        raise RuntimeError(
            f"size mismatch for {dest}: got {size:,} bytes, expected {expected_size:,}"
        )


def count_lines(path):
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            n += 1
    return n


def verify(data_dir):
    problems = 0
    img_dir = os.path.join(data_dir, "Flicker8k_Dataset")
    jpg_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    print(f"  images: {jpg_count} (expected {EXPECTED_IMAGES})")
    if jpg_count != EXPECTED_IMAGES:
        problems += 1
    token_path = os.path.join(data_dir, "Flickr8k.token.txt")
    caption_lines = count_lines(token_path) if os.path.isfile(token_path) else 0
    print(f"  captions: {caption_lines} (expected {EXPECTED_CAPTIONS})")
    if caption_lines != EXPECTED_CAPTIONS:
        problems += 1
    for name, expected in SPLITS:
        path = os.path.join(data_dir, name)
        actual = count_lines(path) if os.path.isfile(path) else 0
        print(f"  {name}: {actual} (expected {expected})")
        if actual != expected:
            problems += 1
    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the Flickr8k dataset (jbrownlee mirror)."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Destination directory (default: <repo>/data/flickr8k)",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Do not delete the zip archives after extraction",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir or os.path.join(repo_root(), "data", "flickr8k"))

    if already_installed(data_dir):
        print(f"Flickr8k already installed at {data_dir}.")
        return

    os.makedirs(data_dir, exist_ok=True)
    print(f"Installing Flickr8k into {data_dir}")

    for name, size in FILES:
        download(f"{BASE_URL}/{name}", os.path.join(data_dir, name), size)

    for name, _ in FILES:
        zip_path = os.path.join(data_dir, name)
        print(f"  extracting {name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)

    if not args.keep_zips:
        for name, _ in FILES:
            os.remove(os.path.join(data_dir, name))

    print("Verifying dataset:")
    if verify(data_dir) == 0:
        print("Done. Flickr8k is ready to use.")
    else:
        print("ERROR: verification failed; check the counts above.")
        sys.exit(1)


if __name__ == "__main__":
    main()