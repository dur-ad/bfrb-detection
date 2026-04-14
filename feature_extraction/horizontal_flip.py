"""
Horizontally flip all videos inside subfolders of a given root folder.

Structure expected:
    to_flip/
        folder1/
            video1.mp4
            video2.mp4
        folder2/
            video3.mp4
        ...

Output:
    flipped/
        folder1/
            video1_flipped.mp4
            video2_flipped.mp4
        folder2/
            video3_flipped.mp4
        ...
"""

import os
import subprocess
import sys


# ── Configuration ─────────────────────────────────────────────────────────────

# Path to the root folder that contains the subfolders with videos.
# Can be passed as a command-line argument or entered interactively.
if len(sys.argv) > 1:
    ROOT_FOLDER = sys.argv[1]
else:
    ROOT_FOLDER = input("Enter the path to your folder: ").strip().strip('"')

# Supported video extensions (case-insensitive)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}

# ──────────────────────────────────────────────────────────────────────────────


def flip_videos(root_folder: str) -> None:
    root_folder = os.path.abspath(root_folder)

    if not os.path.isdir(root_folder):
        print(f"[ERROR] Folder not found: {root_folder}")
        sys.exit(1)

    # Output folder sits next to the root folder
    parent_dir = os.path.dirname(root_folder)
    output_root = os.path.join(parent_dir, "flipped")

    print(f"Source : {root_folder}")
    print(f"Output : {output_root}")
    print()

    total_processed = 0
    total_skipped = 0

    # Walk only one level deep (immediate subfolders)
    for subfolder_name in sorted(os.listdir(root_folder)):
        subfolder_path = os.path.join(root_folder, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue  # skip loose files at root level

        output_subfolder = os.path.join(output_root, subfolder_name)
        os.makedirs(output_subfolder, exist_ok=True)

        print(f"📂  {subfolder_name}")

        for filename in sorted(os.listdir(subfolder_path)):
            name, ext = os.path.splitext(filename)

            if ext.lower() not in VIDEO_EXTENSIONS:
                continue  # skip non-video files

            input_path = os.path.join(subfolder_path, filename)
            output_filename = f"{name}_flipped{ext}"
            output_path = os.path.join(output_subfolder, output_filename)

            if os.path.exists(output_path):
                print(f"   ⚠️  Already exists, skipping: {output_filename}")
                total_skipped += 1
                continue

            print(f"   🎬  {filename}  →  {output_filename}", end="", flush=True)

            # ffmpeg command: hflip filter, copy audio stream, quiet output
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-vf", "hflip",        # horizontal flip
                "-c:a", "copy",        # copy audio without re-encoding
                "-y",                  # overwrite without asking
                "-loglevel", "error",  # suppress ffmpeg chatter
                output_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("  ✅")
                total_processed += 1
            else:
                print("  ❌  FAILED")
                print(f"       {result.stderr.strip()}")
                total_skipped += 1

        print()

    print("─" * 50)
    print(f"Done!  Flipped: {total_processed}  |  Skipped/Failed: {total_skipped}")


if __name__ == "__main__":
    flip_videos(ROOT_FOLDER)
