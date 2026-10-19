"""
Exercise 3: basic preprocessing for the Amazon reviews corpus.

Extends the AmazonCorpusReader of exercise 2.

Arguments:
    --corpus-dir   root folder of the corpus (default: data/corpus)
    --category     describe only these categories (default: all)

Usage:
    python pln_p1_7462_07_e3.py
    python pln_p1_7462_07_e3.py --corpus-dir data/corpus --category positive
    python pln_p1_7462_07_e3.py --help
"""

import argparse
import html
import json
import os
import re
import time
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.corpus.reader import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

CAT_PATTERN = r"(negative|neutral|positive)/.*"

#patterns used to clean the text
#re.compile() turns a regular expression into a pattern object.
#It is done once so the same pattern is reused for every review.
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)          # <br>, <br /> spaces in HTML
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")                # any other HTML tag
MARKUP_RE = re.compile(r"\[\[[^\]]*\]\]")                # [[VIDEOID:..]], [[ASIN:..]]
URL_RE = re.compile(r"(https?://|www\.)\S+")
EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoticons, symbols, pictographs, flags
    "\u2600-\u27BF"           # misc symbols and dingbats
    "\u2B00-\u2BFF"           # stars, arrows
    "\uFE0F\u200D"            # variation selector, zero-width joiner
    "]+"
)
REPEAT_LETTER_RE = re.compile(r"([a-zA-Z])\1{2,}")       # soooo -> soo
REPEAT_PUNCT_RE = re.compile(r"([!?.,;:\-*_~=#])\1+")    # !!!! -> !
HAS_ALNUM_RE = re.compile(r"[a-z0-9]")

# Negations are stopwords in NLTK, but we don't want them to be deleted because they are important for sentiment analysis
NEGATIONS = {"no", "not", "nor"}


def clean_text(text):
    """
    Clean the raw text of a review.

    Paragraph breaks are kept as a blank line so that paragraphs()
    splits the text.

    Parameters:
    - text: str, raw text of the review.

    Returns:
    - str, cleaned text.
    """
#    pattern.sub(replacement, text) searches the text for every part
#    that matches the pattern and replaces it with the replacement.
#    It returns the new text; the original is not modified.
#        BR_RE.sub("\n", "Good.<br />Fast.")   ->   "Good.\nFast."
    text = html.unescape(text or "")          # &amp; -> &, &#34; -> ", html tags
    text = BR_RE.sub("\n", text)              # line breaks written as HTML
    text = TAG_RE.sub(" ", text)
    text = MARKUP_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = REPEAT_LETTER_RE.sub(r"\1\1", text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"[ \t\r\f\v\xa0]+", " ", text)   # extra spaces
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)          # one blank line at most
    return text.strip()


class AmazonCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    Reader for the Amazon reviews corpus built in exercise 1.
    """

    def __init__(self, root="data/corpus", fileids=r".*\.json", encoding="utf8", **kwargs):
        if not any(k.startswith("cat_") for k in kwargs):
            kwargs["cat_pattern"] = CAT_PATTERN
        abs_root = os.path.abspath(root)
        if abs_root not in nltk.data.path:
            nltk.data.path.append(abs_root) #the corpus root has to be registered here so that nltk can find it
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding=encoding)
        self._stopwords = set(stopwords.words("english")) - NEGATIONS

    def _resolve(self, fileids, categories):
            """Return the list of fileids selected by fileids or categories."""
            # Combines category-based filtering with standard fileid lookup
            if fileids is not None:
                if isinstance(fileids, str):
                    fileids = [fileids]
                return fileids
            return self.fileids(categories) #if categories is None, returns all the files in the corpus


    def _paragraphs(self, text):
        """Split a cleaned text into its non-blank paragraphs."""
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _sentences(self, text):
        """Split a cleaned text into its non-blank sentences."""
        sents = []
        for par in self._paragraphs(text):
            for s in nltk.sent_tokenize(par):
                s = s.strip()
                if s:
                    sents.append(s)
        return sents

    def _words(self, text, remove_stopwords=False):
        """
        Split a cleaned text into lowercase tokens.

        Tokens without any letter or digit (punctuation, symbols) are
        removed. Stopwords are removed only if remove_stopwords is True.
        """
        tokens = []
        for sent in self._sentences(text):
            for tok in nltk.word_tokenize(sent):
                tok = tok.lower()
                if not HAS_ALNUM_RE.search(tok): #remove tokens without any letter or digit
                    continue
                if remove_stopwords and tok in self._stopwords:
                    continue
                tokens.append(tok)
        return tokens


    def reviews(self, fileids=None, categories=None):
        """
        Yield each review as a dict with all its fields.

        Empty files and files with invalid JSON are skipped with a warning.
        """
        for fileid in self._resolve(fileids, categories):
            with self.open(fileid) as f:
                content = f.read()
            try:
                review = json.loads(content)
            except json.JSONDecodeError:
                print(f"WARNING: skipped invalid file {fileid} ({len(content)} chars)")
                continue
            yield review

    def ratings(self, fileids=None, categories=None):
        """Yield the rating of each review."""
        for review in self.reviews(fileids, categories):
            yield review.get("rating")

    def texts(self, fileids=None, categories=None, clean=True):
        """Yield the text of each review (raw text if clean is False)."""
        for review in self.reviews(fileids, categories):
            text = review.get("text") or ""
            yield clean_text(text) if clean else text

    def paragraphs(self, fileids=None, categories=None):
        """Yield the list of paragraphs of each review."""
        for text in self.texts(fileids, categories):
            yield self._paragraphs(text)

    def sentences(self, fileids=None, categories=None):
        """Yield the list of sentences of each review."""
        for text in self.texts(fileids, categories):
            yield self._sentences(text)

    def words(self, fileids=None, categories=None, remove_stopwords=False):
        """Yield the list of tokens of each review."""
        for text in self.texts(fileids, categories):
            yield self._words(text, remove_stopwords)


    def describe(self, fileids=None, categories=None):
        """
        Compute the statistics of the corpus in a single pass.

        Parameters:
        - fileids / categories

        Returns:
        - dict with the number of documents, size, review counts,
          rating distribution, products, users, paragraphs, sentences,
          tokens and vocabulary (with and without stopwords).
        """
        start = time.time()
        selected = self._resolve(fileids, categories)

        per_category = Counter()
        rating_dist = Counter()
        tokens = Counter()
        with_rating = with_text = with_both = 0
        n_pars = n_sents = 0
        products, users = set(), set()
        size_bytes = 0

        n_docs = invalid = 0
        for fileid in selected:
            review = next(self.reviews(fileid), None)
            if review is None:          # empty or invalid file
                invalid += 1
                continue
            n_docs += 1
            size_bytes += os.path.getsize(self.abspath(fileid))
            per_category.update(self.categories(fileid))
            review = next(self.reviews(fileid))

            rating = review.get("rating")
            text = clean_text(review.get("text"))
            if rating is not None:
                with_rating += 1
                rating_dist[rating] += 1
            if text:
                with_text += 1
            if rating is not None and text:
                with_both += 1
            if review.get("parent_asin"):
                products.add(review["parent_asin"])
            if review.get("user_id"):
                users.add(review["user_id"])

            n_pars += len(self._paragraphs(text))
            n_sents += len(self._sentences(text))
            tokens.update(self._words(text))

        n_docs = len(selected)
        n_tokens = sum(tokens.values())
        content = {w: c for w, c in tokens.items() if w not in self._stopwords}
        n_content = sum(content.values())

        return {
            "documents": n_docs,
            "documents_per_category": dict(per_category),
            "size_mb": round(size_bytes / 1e6, 2),
            "reviews_with_rating": with_rating,
            "reviews_with_text": with_text,
            "reviews_with_rating_and_text": with_both,
            "rating_distribution": dict(sorted(rating_dist.items())),
            "products": len(products),
            "users": len(users),
            "paragraphs": n_pars,
            "sentences": n_sents,
            "tokens": n_tokens,
            "vocabulary": len(tokens),
            "tokens_no_stopwords": n_content,
            "vocabulary_no_stopwords": len(content),
            "tokens_per_document": round(n_tokens / n_docs, 2) if n_docs else 0,
            "seconds": round(time.time() - start, 2),
        }


def main():
    """
    Parse command-line arguments and print the description of the corpus.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", default="data/corpus",
                        help="root folder of the corpus")
    parser.add_argument("--category", nargs="+", default=None,
                        choices=["positive", "neutral", "negative"],
                        help="describe only these categories")
    args = parser.parse_args()

    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    reader = AmazonCorpusReader(root=args.corpus_dir)
    for key, value in reader.describe(categories=args.category).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
