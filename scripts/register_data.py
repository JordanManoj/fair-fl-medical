"""Register an existing Fed-Heart-Disease download with FLamby (no re-download).

Checks the four UCI files against the MD5 hashes in FLamby's download script,
then writes FLamby's dataset config so FedHeartDisease() finds the folder.

Usage:
  python scripts/register_data.py path/to/heart_folder
"""
import hashlib
import sys
from pathlib import Path

from flamby.utils import create_config, write_value_in_config

# from flamby/datasets/fed_heart_disease/dataset_creation_scripts/download.py
MD5 = {
    "processed.cleveland.data": "2d91a8ff69cfd9616aa47b59d6f843db",
    "processed.hungarian.data": "22e96bee155b5973568101c93b3705f6",
    "processed.switzerland.data": "9a87f7577310b3917730d06ba9349e20",
    "processed.va.data": "4249d03ca7711e84f4444768c9426170",
}


def main(folder):
    folder = Path(folder).resolve()
    for name, expected in MD5.items():
        f = folder / name
        if not f.exists():
            sys.exit(f"Missing {f}")
        if hashlib.md5(f.read_bytes()).hexdigest() != expected:
            sys.exit(f"MD5 mismatch for {f}")
    _, config_file = create_config(str(folder), False, "fed_heart_disease")
    write_value_in_config(config_file, "dataset_path", str(folder))
    write_value_in_config(config_file, "download_complete", True)
    write_value_in_config(config_file, "preprocessing_complete", True)
    print(f"Registered {folder} (4 files, MD5 verified)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
