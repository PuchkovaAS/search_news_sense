import string
from collections import Counter
import nltk
from lxml import etree
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# import nltk
# nltk.download()

def get_tfid(titles, dataset):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset)
    terms = vectorizer.get_feature_names()
    # print(tfidf_matrix)
    # print(terms[11])
    from itertools import count
    for index, title in enumerate(titles):
        lst = tfidf_matrix.toarray()[index]
        new_lst = [x[0] for x in sorted(enumerate(lst), key=lambda x: x[1])][::-1]
        # print(new_lst)
        print(f"{title}:")

        for el in new_lst[0:5]:
            print(terms[el], end=' ')
        print('\n', end='\n')


def get_title(elem):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    punct = list(string.punctuation)
    all_word = []
    title = elem[0].text

    # tokkenization
    all_word.extend(word_tokenize(elem[1].text.lower()))

    # lemmanization
    all_word = [lemmatizer.lemmatize(word) for word in all_word]

    # getout stopword
    all_word = [word for word in all_word if word not in stop_words and word not in punct]

    # get nonse
    nonse_list = sorted([word for word in all_word if nltk.pos_tag([word])[0][1] == 'NN'])

    # get_tfid(' '.join(nonse_dict))
    return title, ' '.join(nonse_list)
    # return title, ' '.join([word for word, count in nonse_dict][0:5])

titles = []
nonse_lists = []
xml_file = "news.xml"
root = etree.parse(xml_file).getroot()[0]
for elem in root:
    title, nonse_list = get_title(elem)
    titles.append(title)
    nonse_lists.append(nonse_list)


    # title, words = get_title(elem)
    # print(f"{title}:", words, sep='\n')
    # print()
get_tfid(titles=titles, dataset=nonse_lists)
