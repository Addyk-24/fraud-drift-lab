import subprocess
import sys
import zipfile

from src import config

NEEDED = ["train_transaction.csv", "train_identity.csv"]


def main():
    config.ensure_dirs()
    if all((config.RAW_DIR / f).exists() for f in NEEDED):
        print("Raw files already present in", config.RAW_DIR)
        return
    cmd = [sys.executable, "-m", "kaggle", "competitions", "download",
           "-c", "ieee-fraud-detection", "-p", str(config.RAW_DIR)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    zpath = config.RAW_DIR / "ieee-fraud-detection.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            for f in NEEDED:
                z.extract(f, config.RAW_DIR)
        zpath.unlink()
    print("Done. Files in", config.RAW_DIR)


if __name__ == "__main__":
    main()
