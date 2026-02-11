#!/usr/bin/env python3
"""Strip a given string (text) from all YAML files in the same folder."""

import argparse
from pathlib import Path


def strip_text_from_yamls(root_dir: Path, text_to_strip: str, dry_run: bool = False) -> int:
    """Remove the given text from all YAML files in root_dir. Returns count of modified files."""
    root_dir = Path(root_dir).resolve()
    modified = 0

    for pattern in ("*.yaml", "*.yml"):
        for yaml_path in root_dir.glob(pattern):
            content = yaml_path.read_text(encoding="utf-8")
            if text_to_strip in content:
                new_content = content.replace(text_to_strip, "")
                if dry_run:
                    print(f"Would modify: {yaml_path}")
                else:
                    yaml_path.write_text(new_content, encoding="utf-8")
                    print(f"Modified: {yaml_path}")
                modified += 1

    return modified


def main():
    parser = argparse.ArgumentParser(description="Strip text from all YAML files in the same folder")
    parser.add_argument("text", help="The string/text to remove from YAML files")
    parser.add_argument(
        "-d", "--directory",
        default=".",
        help="Directory with YAML files (default: current directory)",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Show what would be modified without changing files",
    )
    args = parser.parse_args()

    count = strip_text_from_yamls(Path(args.directory), args.text, args.dry_run)
    if args.dry_run:
        print(f"\nWould modify {count} file(s). Run without -n to apply.")
    else:
        print(f"\nModified {count} file(s).")


if __name__ == "__main__":
    main()
