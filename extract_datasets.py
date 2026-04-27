import zipfile
import gzip
import os
import shutil

EMNIST_SPLITS = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]

zip_path = "gzip.zip"

print("Extracting zip...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(os.path.join("datasets", "_emnist_tmp"))

tmp = os.path.join("datasets", "_emnist_tmp", "gzip")

for split in EMNIST_SPLITS:
    out = os.path.join("datasets", split)
    os.makedirs(out, exist_ok=True)
    for fname in os.listdir(tmp):
        if fname.startswith(f"emnist-{split}-") and fname.endswith(".gz"):
            gz_path = os.path.join(tmp, fname)
            out_name = fname[:-3]
            out_path = os.path.join(out, out_name)
            print(f"Extracting {out_name}...")
            with gzip.open(gz_path, "rb") as f_in:
                with open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

shutil.rmtree(os.path.join("datasets", "_emnist_tmp"))
print("\nDone! Datasets are in datasets/")