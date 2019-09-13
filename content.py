import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from collections import Counter
from bs4 import BeautifulSoup
from stemming.porter2 import stem
from string import digits
import nltk
import string
import  sklearn
import re
import pdb

stemmer = SnowballStemmer('english')
news = pd.read_csv('./news_articles.csv')
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

cleaned_articles = list(map(clean, contents))

#pdb.set_trace()

article_vocab = enumerate(cleaned_articles)

total_words = []
for i in range(0, len(cleaned_articles)):
    tokens = nltk.word_tokenize(cleaned_articles[i])
    for word in tokens:
        total_words.append(word)

counts = set(total_words)

stops_removed = [i for i in counts if i not in stops]

final_vocab = {j:i for i,j in enumerate(stops_removed)}

tf_idf = TfidfVectorizer(vocabulary=final_vocab, min_df=1)

article_vocabulary_matrix = tf_idf.fit_transform(cleaned_articles)

lda = LatentDirichletAllocation(n_components=n_topics, max_iter=1, random_state=0)

Lda_articlemat = lda.fit_transform(article_vocabulary_matrix)

wordtokens_article = [word.split() for word in cleaned_articles] 

def user_profiler(wordtokens,article_read,article_time):
    user_profile = []
    wordPer_second = 5
    for i in range(len(wordtokens)):
        average_time = (len(wordtokens[i])/wordPer_second) #length of wordtokslist by wps gives us avg time to read the article
        user_interest_timevalue = article_time[i]/average_time  #article_times divide by avg times of each article
        user_profile_generate = (article_read[i]*user_interest_timevalue)          #Ldamatrix[] * user_interest_time calculated
        user_profile.append(user_profile_generate)
   # pdb.set_trace()
    return sum(user_profile)

userProfile_One = user_profiler([wordtokens_article[600],wordtokens_article[99],wordtokens_article[120]],
                         [Lda_articlemat[600],Lda_articlemat[99],Lda_articlemat[120]],
                         [120,60,30])

userProfile_Two = user_profiler([wordtokens_article[900],wordtokens_article[500],wordtokens_article[3000]],
                         [Lda_articlemat[900],Lda_articlemat[500],Lda_articlemat[3000]],
                         [111,120,180])

userProfile_Three = user_profiler([wordtokens_article[600],wordtokens_article[4830],wordtokens_article[390]],
                           [Lda_articlemat[600],Lda_articlemat[4830],Lda_articlemat[390]],
                           [200,120,100])

userprofile_List = [userProfile_One,userProfile_Two,userProfile_Three]
#print(userProfile_One)

#normalized_profiles = Normalizer(csr_matrix(userprofile_List))

def similar():
    n = []
    for profiles in userprofile_List:
        user_preferred_articles = cosine_similarity(profiles.reshape(1,-1),Lda_articlemat)
        a = np.argsort(user_preferred_articles).flatten()[::-1][:no_of_recommends]
        n.append(a)
    return n

similarityscore = similar()

counter = 0
for i in similarityscore:
    print('\n')
    print('Recommended Articles :')
    print('\n')
    print(news['Title'][i])

pdb.set_trace()
