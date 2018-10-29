import nltk
import pandas as pd
import re
import string
from nltk.tokenize import RegexpTokenizer

ps = nltk.PorterStemmer()


def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text


def tokensization(corpus):
    corpus.dropna()
    text_nopunct = []
    tokenizer = RegexpTokenizer(r'\w+')
    for row in corpus['ApplicationSummary']:
        text_nopunct.append(tokenizer.tokenize(row))
    dat1 = pd.DataFrame({'tokens': text_nopunct})
    corpus = corpus.join(dat1)
    return corpus


pd.set_option('display.max_colwidth', 200)
stop = nltk.corpus.stopwords.words('english')
french_stop = nltk.corpus.stopwords.words('french')
data = pd.read_csv("./data/NSERC_GRT_FYR2017_AWARD.csv", encoding="ISO-8859-1")

data = data.dropna()
data = tokensization(data)


def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

def clean_text_english(text):
    stop = nltk.corpus.stopwords.words('english')
    text = [word for word in text if word.lower() not in stop]
    return text


def clean_text_french(text):
    stop = nltk.corpus.stopwords.words('french')
    text = [word for word in text if word.lower() not in stop]
    return text

data = data.dropna()
data['body_text_nostop_english'] = data['tokens'].apply(lambda x: clean_text_english(x))
data['body_text_nostop_french'] = data['body_text_nostop_english'].apply(lambda x: clean_text_french(x))

data['body_text_stemmed'] = data['body_text_nostop_french'].apply(lambda x: stemming(x))

print(data['body_text_stemmed'].head())
print(data['body_text_nostop_french'].head())

data.to_csv("NSERC_GRT_FYR2017_AWARD_STEMED.csv", encoding='ISO-8859-1')
