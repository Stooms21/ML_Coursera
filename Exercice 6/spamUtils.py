import re
import numpy as np
from nltk.stem import PorterStemmer


def getVocabList():
    fid = open('../Data/vocab.txt', 'r')
    vocab = fid.readlines()
    fid.close()
    for i in range(len(vocab)):
        vocab[i] = re.sub('[0-9]+', '', vocab[i])
        vocab[i] = re.sub('\t', '', vocab[i])
        vocab[i] = re.sub('\n', '', vocab[i])
    return vocab


def processEmail(email_contents):
    # PROCESSEMAIL preprocesses a the body of an email and
    # returns a list of word_indices
    # word_indices = PROCESSEMAIL(email_contents) preprocesses
    # the body of an email and returns a list of indices of the
    # words contained in the email.
    #

    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)

    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)

    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)

    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]

    # Stem the email contents word by word
    stemmer = PorterStemmer()
    processed_email = []
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)

        if len(word) < 1:
            continue

        # Look up the word in the dictionary and add to word_indices if found
        # ====================== YOUR CODE HERE ======================

        try:
            word_indices.append(vocabList.index(word))
        except ValueError:
            pass

        # =============================================================

    return word_indices


def emailFeatures(word_indices):
    n = 1899
    x = np.zeros(n)
    x[word_indices] = 1
    return x
