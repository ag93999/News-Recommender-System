import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from nltk import sent_tokenize, word_tokenize, pos_tag
from collections import Counter
from stemming.porter2 import stem
from bs4 import BeautifulSoup
from string import digits
import nltk
import string
import  sklearn
import re
import pdb

stemmer = SnowballStemmer('english')
news = pd.read_csv('./news_articles.csv')
news.head()
stops = set(stopwords.words('english'))

no_of_recommends = 5
n_topics = 8

news = news[['Article_Id','Title','Content']].dropna()
contents = news["Content"].tolist()
title = news['Title']
article_id = news['Article_Id']

regex = r'\w+'
stop = set(stopwords.words('english'))- set({'not','didn','shouldn','haven','won','weren','wouldn','hasn','couldn','ain','needn','mightn','don','nor','isn','shan','no','wasn','mustn','hadn'})
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
cleanr = re.compile('<.*?>')

def clean(doc):
    remove_digits = str.maketrans('', '', digits)
    doc = doc.translate(remove_digits)
    cleantext = BeautifulSoup(doc, "html.parser").text
    doc = re.sub(cleanr, ' ', doc)
    doc = doc.replace("<div>"," ")
    doc = doc.replace("</div>"," ")
    doc = doc.replace(".</div>"," ")
    doc = doc.replace("<br />"," ")
    doc = doc.replace("."," ")
    doc = doc.replace(":"," ")
    doc = doc.replace(","," ")
    doc = doc.replace("_"," ")
    doc = doc.replace('-', ' ')
    doc = doc.replace('(', ' ')
    doc = doc.replace(')', ' ')
    doc = doc.replace('#', ' ')
    doc = doc.replace('/', ' ')
    doc = doc.replace(" div "," ")
    doc = doc.replace(" br ", " ")
    doc = doc.replace("nbsp"," ")
    doc = doc.replace("ndash"," ")
    doc = doc.replace("&rsquo;", ' ')
    doc = doc.replace("&trade;", ' ')
    doc = re.sub(r"\&([^;.]*);", " ", doc)
    doc = re.sub(r"([0-9]+)-([0-9]+)", " ", doc)
    doc = re.sub(r"\d", " ", doc)
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    punc_free = re.sub(r"\b\d+\b"," ",punc_free)
    words = word_tokenize(punc_free)
    lemmatized_words = [lemma.lemmatize(word) for word in words]
    #stemmed_word = [stem(w) for w in lemmatized_words]
    #stemmed_word = [stem(w) for w in words]
    finallist = []
    for ch in lemmatized_words:
        if len(ch) > 2 and len(ch) < 13 and ch.encode('utf-8').isalnum() == True and bool(re.search(r'\d', ch)) == False:
            try:
                finallist.append(stem(vocab_mapper[ch]))
            except:
                finallist.append(stem(ch))
    final = " ".join(finallist)
    return final

def clean_tokenize(document):
    document = re.sub('[^\w_\s-]',' ',document)
    tokens  = nltk.word_tokenize(document)
    cleaned_article = ' '.join([stemmer.stem(item) for item in tokens])   #stemming the tokenized corpus
    return cleaned_article

cleaned_articles = list(map(clean,contents))
pdb.set_trace()
article_vocab = { }

article_vocab = enumerate(cleaned_articles)

total_words = []

for i in range(0, len(cleaned_articles)):
    tokens = nltk.word_tokenize(cleaned_articles[i])

    for w in tokens:
        total_words.append(w)
counts = Counter(total_words)

vocab = {j:i for i,j in enumerate(counts.keys())}

stops_removed = [i for i in vocab.keys() if i not in stops]

final_vocab = {j:i for i,j in enumerate(stops_removed)}

tf_idf = TfidfVectorizer(vocabulary=final_vocab,min_df=1)

article_vocabulary = tf_idf.fit_transform(cleaned_articles)

lda = LatentDirichletAllocation(n_components=n_topics,max_iter=1,random_state=0)

Lda_articlemat = lda.fit_transform(article_vocabulary)

wordtokens_article = [word.split() for word in cleaned_articles]

existing_users = np.random.random_sample(size=(10000,8)) #we take any number of users

new_user = np.random.random_sample(size=(1,8)) #we take a user to recommend him according to collaborative filtering by using cosine similiarty

Similarity_Score = cosine_similarity(existing_users,new_user) #now we get our similar user score from our existing users and new user

top_similars_users = np.argsort(Similarity_Score,axis=0)[::-1][:5]

top_users= existing_users[top_similars_users]  #picking our top 5 similar user profiles out of existing users

avg_user_profile = np.max(top_users,axis=0)   # we take mean to get all the average or all the existing users


sim_articles = cosine_similarity(avg_user_profile,Lda_articlemat) #now we find our similar articles according to our avg u.p.

interested_articles = np.argsort(sim_articles)[::-1]  #we fetch the indexes of our top 5 similar articles according to user interest

for i in [interested_articles[0]] :
    print('Recommended-Articles :')
    print('\n')
    print(news['Title'][i][:20])

pdb.set_trace()
