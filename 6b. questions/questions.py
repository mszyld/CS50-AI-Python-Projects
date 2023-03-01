import nltk
nltk.download('stopwords')
import sys
import os
import string
import numpy as np

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    ans = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            print(filename)
            with open(os.path.join(directory, filename), encoding="utf8") as f:
                ans[filename] = str(f.readlines())
    return ans


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = [w.lower() for w in nltk.word_tokenize(document)]
    return [w for w in words if w not in string.punctuation and w not in nltk.corpus.stopwords.words("english")]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    appears_in = {}
    for doc, words in documents.items():
        for word in words:
            if word in appears_in:
                appears_in[word].add(doc)
            else:
                appears_in[word] = {doc}
    ans = {}
    for word in appears_in:
        ans[word] = np.log( len(documents) / len(appears_in[word]) )
    return ans


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf_sum = dict.fromkeys(files.keys(),0)
    for word in query:
        for file in files:
            tf_idf_sum[file] += files[file].count(word) * idfs[word]
    ans = list(files.keys()).copy()
    return sorted(ans, key=lambda f: tf_idf_sum[f], reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf_sum = dict.fromkeys(sentences.keys(),0)
    query_density = dict.fromkeys(sentences.keys(),0)
    for word in query:
        for sentence, words in sentences.items():
            if word in words:
                idf_sum[sentence] += idfs[word]
                query_density[sentence] += 1 / len(words)
    ans = list(sentences.keys()).copy()
    return sorted(ans, key=lambda s: (idf_sum[s], query_density[s]), reverse=True)[:n]


if __name__ == "__main__":
    main()
