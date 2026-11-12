"""Verify the bundled data using SHA-256; no third-party packages required."""
from pathlib import Path
import hashlib

def main():
    root = Path(__file__).resolve().parent
    errors, count = [], 0
    for line in (root / "data/checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            errors.append(relative + " missing")
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected != actual:
            errors.append(relative + " changed")
        count += 1
    print(f"Checked {count} data files; {len(errors)} errors.")
    for error in errors:
        print(error)
    return 1 if errors else 0

if __name__ == "__main__":
    raise SystemExit(main())
