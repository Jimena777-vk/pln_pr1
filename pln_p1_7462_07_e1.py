"""
Exercise 1: download and prepare the Amazon reviews corpus.

Downloads a review file, labels each review by rating, and saves it as
a JSON file inside its class folder:

    <corpus_dir>/
        positive/  <source>_<n>.json   (rating 4-5)
        neutral/  <source>_<n>.json   (rating 3)
        negative/  <source>_<n>.json   (rating 1-2)

Arguments:
    --dataset          one or more of: test, games, clothing
                        (default: test)
    --data-dir         base folder; raw/ and corpus/ are created inside
                        (default: data)
    --max-per-class    max. reviews saved per class and dataset
                        (default: no limit)

Usage:
    python pln_p1_7462_07_e1.py --dataset test
    python pln_p1_7462_07_e1.py --dataset games clothing --max-per-class 20000
    python pln_p1_7462_07_e1.py --data-dir my_folder --dataset test
    python pln_p1_7462_07_e1.py --help
"""


import argparse
import gzip
import json
import urllib.request
from pathlib import Path

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

DATASETS = {
    "test": "Subscription_Boxes", 
    "clothing": "Amazon_Fashion",
    "games": "Video_Games",
}


REVIEW_FIELDS = ["rating", "title", "text", "asin", "parent_asin",
                 "user_id", "verified_purchase", "helpful_vote"]


def review_url(category):
    """
    Build the download URL of the review file for an Amazon category.

    Parameters:
    - category: str, Amazon category name as it appears in the dataset
      (e.g. "Video_Games", "Amazon_Fashion").

    Returns:
    - str, URL of the compressed reviews file (.jsonl.gz).
    """
    return f"{BASE}/review_categories/{category}.jsonl.gz"


def rating_to_label(rating):
    """
    1-2 -> negative
    3 -> neutral
    4-5 -> positive
    Returns None otherwise"""
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return None
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def download_file(url, dest):
    """
    Download a file from a URL to a local path, unless it already exists.

    The file is kept compressed (.gz). It is first saved with a ".part"
    suffix and renamed only when the download finishes, so an interrupted
    download never leaves a corrupt file with the final name.

    Parameters:
    - url: str, URL of the file to download.
    - dest: str or Path, local path where the file will be saved.

    Returns:
    - Path, the path of the downloaded (or already existing) file.
    """
    dest = Path(dest) #transform into a Path() object
    if dest.exists():
        print(f"{dest.name} Already exists")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Download -{url}")
    tmp = dest.with_suffix(dest.suffix + ".part") #creates temporal file
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)  # rename once the download is done
    print(f"ok - {dest}")
    return dest


def iter_jsonl_gz(path):
    """
    Read a compressed JSON Lines file (.jsonl.gz) one record at a time.

    The file is never extracted to disk nor fully loaded into memory.

    Parameters:
    - path: str or Path, path to the .jsonl.gz file.

    Yields:
    - dict, one parsed JSON object (e.g. one review) per line.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: #blank lines are skipped
                yield json.loads(line)


def build_reviews(gz_path, corpus_dir, source, max_per_class):
    """
    Organize a reviews file in separate folders per class.

    Each review is labeled as: 
    - "positive" (rating 4-5)
    - "neutral" (rating 3)
    - "negative" (rating 1-2)
    
    It is saved to <corpus_dir>/<label>/. Reviews without
    a valid rating or without text are skipped. Reading stops once every
    class has reached max_per_class reviews.

    Parameters:
    - gz_path: str or Path, path to the .jsonl.gz reviews file.
    - corpus_dir: str or Path, root folder of the corpus.
    - source: str, short dataset name, used as prefix of the file names
      (e.g. "games", "clothing").
    - max_per_class: int or None, maximum number of reviews saved per
      class (None means no limit).

    Returns:
    - tuple (set, dict): the parent_asin (Parent ID of the product) of the products found in the
      saved reviews, and the number of reviews saved per class.
    """
    corpus_dir = Path(corpus_dir)
    for label in ("positive", "neutral", "negative"):
        (corpus_dir / label).mkdir(parents=True, exist_ok=True) #creates the three folders

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    skipped = 0 #count the skipped reviews
    products = set() #avoid repeating the same products

    for review in iter_jsonl_gz(gz_path):
        label = rating_to_label(review.get("rating"))
        text = (review.get("text") or "").strip()
        #skip the products with no label or review
        if label is None or not text:
            skipped += 1
            continue
        if max_per_class is not None and counts[label] >= max_per_class:
            if all(c >= max_per_class for c in counts.values()): #break if ALL classes reach the limit
                break
            continue

        counts[label] += 1
        doc = {k: review.get(k) for k in REVIEW_FIELDS}
        doc["label"] = label
        doc["source"] = source
        out = corpus_dir / label / f"{source}_{counts[label]:07d}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        products.add(review.get("parent_asin"))

    print(f"[{source}] saved: {counts} | skipped: {skipped}")
    return products, counts



def main():
    """
    Parse command-line arguments and build the corpus for each dataset.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", default=["test"],
                    choices=list(DATASETS), help="datasets to process")
    parser.add_argument("--data-dir", default="data",
                        help="base folder (raw/ and corpus/ are created inside)")
    parser.add_argument("--max-per-class", type=int, default=1000,
                help="max. reviews per class and dataset (empty = no limit)")

    args = parser.parse_args()

    raw_dir = Path(args.data_dir) / "raw"
    corpus_dir = Path(args.data_dir) / "corpus"
    max_per_class = args.max_per_class or None

    for name in args.dataset:
        category = DATASETS[name]
        gz = download_file(review_url(category), raw_dir / f"{category}.jsonl.gz")
        _, _ = build_reviews(gz, corpus_dir, name, max_per_class)


if __name__ == "__main__":
    main()