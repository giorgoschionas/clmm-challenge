"""Rebuild deterministic participant downloads after a reviewed release."""
import argparse
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "hackathon-frontend/public/clmm")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ["README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "UPSTREAM.json",
             "pyproject.toml", "MANIFEST.in", "requirements.txt"]
    sources = [ROOT / name for name in names]
    sources += sorted((ROOT / "clmm_challenge").glob("*.py"))
    sources += sorted((ROOT / "examples").glob("*.py"))
    sources += sorted((ROOT / "tests").glob("*.py"))
    with ZipFile(args.output / "clmm-challenge.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sources:
            info = ZipInfo("clmm-challenge/" + path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    shutil.copyfile(ROOT / "README.md", args.output / "README.md")
    shutil.copyfile(ROOT / "examples" / "submission_template.py", args.output / "submission_template.py")


if __name__ == "__main__":
    main()
