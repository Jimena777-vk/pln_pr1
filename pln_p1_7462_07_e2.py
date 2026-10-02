from nltk.corpus.reader import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
import nltk
import os
import json

class AmazonCorpusReader(CategorizedCorpusReader, CorpusReader):
    def __init__(self, root='data/corpus', fileids=r'.*\.json', encoding='utf8', **kwargs):
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding=encoding)

    def _resolve(self, fileids, categories):
        # Combines category-based filtering with standard fileid lookup
        if fileids is not None:
            if isinstance(fileids, str):
                fileids = [fileids]
            return fileids
        return self.fileids(categories)

    def _parse_jsonl(self, stream):
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)

    def reviews(self, fileids=None, categories=None):
        fileids = self._resolve(fileids, categories)
        for review in fileids:
            with self.open(review) as f:
                for review in self._parse_jsonl(f):
                    yield review

    def ratings(self, fileids=None, categories=None):
        for review in self.reviews(fileids=fileids, categories=categories):
            yield review['rating']

    def texts(self, fileids=None, categories=None):
        for review in self.reviews(fileids=fileids, categories=categories):
            yield review['text']

    def paragraphs(self, fileids=None, categories=None):
        for text in self.reviews(fileids=fileids, categories=categories):
            yield [p.strip() for p in text.split('\n\n') if p.strip()]

    def sentences(self, fileids=None, categories=None):
        for text in self.texts(fileids=fileids, categories=categories):
            yield nltk.sent_tokenize(text)

    def words(self, fileids=None, categories=None):
        for text in self.texts(fileids=fileids, categories=categories):
            yield nltk.word_tokenize(text)


def main():
    corpus_root = 'data/corpus'
    nltk.download('punkt_tab',quiet=True)

    abs_corpus_root = os.path.abspath(corpus_root)
    if abs_corpus_root not in nltk.data.path:
        nltk.data.path.append(abs_corpus_root)

    cat_pattern = r"(negative|neutral|positive)/.*"
    reader = AmazonCorpusReader(root='data/corpus/', fileids=r'.*\.json', cat_pattern=cat_pattern)
    return reader

if __name__ == "__main__":
    main()