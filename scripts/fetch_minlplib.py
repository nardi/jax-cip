"""Downloads MINLPLib instances in OSiL form with their known solutions.

The files go into the directory in `JAXCIP_MINLPLIB_DIR`, or the given one: `minlplib.solu`,
`instancedata.csv` and an `osil` directory. Without instance names the whole OSiL archive
is downloaded, about 140 MB. Files that exist already are kept.

Usage: python scripts/fetch_minlplib.py [--directory DIRECTORY] [NAME ...]
"""

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

SITE = "https://www.minlplib.org"
"""Where MINLPLib is published."""


def download(url: str, path: Path) -> None:
    """Downloads a file unless it exists, writing it under its name only when complete."""
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as file:
        shutil.copyfileobj(response, file)
    temporary.replace(path)


def main() -> None:
    """Downloads the files named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--directory", type=Path, default=os.environ.get("JAXCIP_MINLPLIB_DIR")
    )
    parser.add_argument("names", nargs="*", help="instances to download")
    arguments = parser.parse_args()
    if arguments.directory is None:
        parser.error("pass --directory or set JAXCIP_MINLPLIB_DIR")
    directory: Path = arguments.directory
    instances = directory / "osil"
    instances.mkdir(parents=True, exist_ok=True)
    for name in ("minlplib.solu", "instancedata.csv"):
        download(f"{SITE}/{name}", directory / name)
    if arguments.names:
        for name in arguments.names:
            download(f"{SITE}/osil/{name}.osil", instances / f"{name}.osil")
        return
    archive = directory / "minlplib_osil.zip"
    download(f"{SITE}/minlplib_osil.zip", archive)
    with zipfile.ZipFile(archive) as files:
        for member in files.infolist():
            name = Path(member.filename).name
            if name.endswith(".osil") and not (instances / name).exists():
                with (
                    files.open(member) as source,
                    (instances / name).open("wb") as file,
                ):
                    shutil.copyfileobj(source, file)


if __name__ == "__main__":
    main()
