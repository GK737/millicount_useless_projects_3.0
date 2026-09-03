"""
Fetch high-resolution millipede (Diplopoda) macro photographs from iNaturalist Open Data API.
Saves images and license/attribution metadata into `raw_images/`.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
from pathlib import Path


def fetch_millipede_images(output_dir="raw_images", limit=20, quality_grade="research"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_file = output_path / "metadata.json"
    existing_meta = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
        except Exception:
            existing_meta = {}

    headers = {
        "User-Agent": "MillipedeDatasetBuilder/1.0 (academic_research_and_segmentation)"
    }

    per_page = min(limit, 50)
    page = 1
    downloaded = 0

    print(f"[*] Querying iNaturalist API for Diplopoda (Taxon ID: 47735, quality: {quality_grade})...")

    while downloaded < limit:
        url = (
            f"https://api.inaturalist.org/v1/observations?"
            f"taxon_id=47735&photos=true&quality_grade={quality_grade}"
            f"&per_page={per_page}&page={page}&order=desc&order_by=votes"
        )

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as res:
                data = json.loads(res.read().decode("utf-8"))
        except Exception as e:
            print(f"[!] Error querying iNaturalist API: {e}")
            break

        results = data.get("results", [])
        if not results:
            print("[*] No more observations found.")
            break

        for obs in results:
            obs_id = obs.get("id")
            taxon_name = obs.get("taxon", {}).get("name", "Diplopoda")
            common_name = obs.get("taxon", {}).get("preferred_common_name", "Millipede")
            user_login = obs.get("user", {}).get("login", "Unknown")

            photos = obs.get("photos", [])
            for idx, photo in enumerate(photos):
                if downloaded >= limit:
                    break

                photo_id = photo.get("id")
                license_code = photo.get("license_code")
                square_url = photo.get("url", "")
                if not square_url:
                    continue

                # Get large image URL
                large_url = square_url.replace("square.", "large.")
                file_ext = Path(square_url).suffix or ".jpg"
                filename = f"inat_{obs_id}_{photo_id}{file_ext}"
                target_file = output_path / filename

                if target_file.exists():
                    # already downloaded
                    continue

                print(f"[{downloaded + 1}/{limit}] Downloading: {filename} ({taxon_name} by @{user_login})...")
                try:
                    img_req = urllib.request.Request(large_url, headers=headers)
                    with urllib.request.urlopen(img_req, timeout=20) as img_res, open(target_file, "wb") as out_f:
                        out_f.write(img_res.read())

                    existing_meta[filename] = {
                        "obs_id": obs_id,
                        "photo_id": photo_id,
                        "taxon_name": taxon_name,
                        "common_name": common_name,
                        "photographer": user_login,
                        "license": license_code,
                        "url": large_url,
                    }
                    downloaded += 1
                    time.sleep(0.5)  # respectful rate limit
                except Exception as dl_err:
                    print(f"    [!] Failed to download {large_url}: {dl_err}")

        page += 1

    # Save updated metadata
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(existing_meta, f, indent=2)

    print(f"\n[+] Finished downloading {downloaded} images to '{output_dir}'.")
    print(f"[+] Metadata saved to '{metadata_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch real millipede macro photos from iNaturalist.")
    parser.add_argument("--limit", type=int, default=10, help="Number of images to fetch (default: 10)")
    parser.add_argument("--output", type=str, default="raw_images", help="Target output directory")
    args = parser.parse_args()

    fetch_millipede_images(output_dir=args.output, limit=args.limit)
