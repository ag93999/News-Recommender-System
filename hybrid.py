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
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from string import digits
import nltk
import string
import  sklearn
import re
import pdb

stemmer = SnowballStemmer('english')
news = pd.read_csv('./news_articles.csv')
stops = set(stopwords.words('english'))

no_of_recommends = 20
n_topics = 10
no_of_clusters = 10
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

cleaned_articles = list(map(clean, contents))


def Topic_Modeller(LDA_matrix):
    
    total_WordVocab = []
    for i in range(0,len(cleaned_articles)) :
        word_tokens = nltk.word_tokenize(cleaned_articles[i])
        for words in word_tokens :
            total_WordVocab.append(words)
        counts = Counter(total_WordVocab)

    vocab = {j:i for i,j in enumerate(counts.keys())}
    stops_removed = [word for word in vocab.keys() if word not in stops]
    Final_VocabDict = {j:i for i,j in enumerate(stops_removed)}
    
    Tfidf = TfidfVectorizer(min_df=1,vocabulary=Final_VocabDict)
    Tfidf_Matrix = Tfidf.fit_transform(cleaned_articles)

    Lda = LatentDirichletAllocation(n_components=n_topics,max_iter=1,random_state=0)
    Lda_articlemat = Lda.fit_transform(Tfidf_Matrix)

    return Lda_articlemat

wordtokens_article = [word.split() for word in cleaned_articles]  #for userprofiles

Lda_articlemat = Topic_Modeller(cleaned_articles)

kmeans = KMeans(n_clusters=no_of_clusters)

clustered_articles_matrix = kmeans.fit_transform(Lda_articlemat)

wordtokens_article = [word.split() for word in cleaned_articles]

def user_profiler(wordtokens,article_read,article_time):
    user_profile = []
    wordPer_second = 5
    for i in range(len(wordtokens)):
        average_time = (len(wordtokens[i])/wordPer_second) #length of wordtokslist by wps gives us avg time to read the article
        user_interest_timevalue = article_time[i]/average_time  #article_times divide by avg times of each article
        user_profile_generate = (article_read[i]*user_interest_timevalue)   #clustered_articles_matrix[] * user_interest_time calculated
        user_profile.append(user_profile_generate)

    return sum(normalize(user_profile))

userProfile_one = user_profiler([wordtokens_article[600],wordtokens_article[99],wordtokens_article[120]],
                             [clustered_articles_matrix[600],clustered_articles_matrix[99],clustered_articles_matrix[120]],
                             [120,60,30])

def Content_Recommends_Calculator(user_profile,clustered_articles_matrix) :
    user_interested_articles = []
    contents_interest_score = []
    user_preffered_articles = cosine_similarity(userProfile_one.reshape(1,-1),clustered_articles_matrix)
    top_articles = np.sort(user_preffered_articles).flatten()[::-1][:10]
    user_interested_articles.append(top_articles)
    content_interest_score = (user_interested_articles[0] * 0.4)
    return content_interest_score

content_recommended = Content_Recommends_Calculator(userProfile_one,clustered_articles_matrix)

existing_users = np.random.random_sample(size=(1000,10))   #we take n existing users
new_user = np.random.random_sample(size=(1,10))            #we take a single new user


def Collaborative_Recommends_Calculator(existing_usr,new_usr) :
    collaborative_interest_score = [ ]
    sorted_collaborative_interest = [ ]
    collaborative_interest_score = cosine_similarity(existing_users,new_user)
    sorted_collaborative_interest = np.argsort(collaborative_interest_score,axis=0)[::-1][:10]
    sorted_collaborative_indexes = existing_users[sorted_collaborative_interest]
    collab_interest = np.mean(sorted_collaborative_indexes.reshape(-1,10),axis=0)
    collaborative_interest_scores = collab_interest*0.6
    return collaborative_interest_scores

collab_recommended = Collaborative_Recommends_Calculator(existing_users,new_user)

def Trends(trending) :
    trends = np.mean(existing_users,axis=0)
    Trending_news = cosine_similarity(trends.reshape(1,10),clustered_articles_matrix)
    top= np.sort(Trending_news)[::-1][0][:10]
    return top

Trending_Articles = Trends(existing_users)*0.3


def Hybrid_Calculator():
    hybrid_interests = np.add(content_recommended,collab_recommended)
    similar_scores = cosine_similarity(hybrid_interests.reshape(1,10),clustered_articles_matrix)
    recommended_article_address =  np.argsort(similar_scores)[::-1]
    return recommended_article_address

hybrid_recommend_indexes =  Hybrid_Calculator() #we get hyrid interest with our variations of 0.4 content based and 0.6 collab based

labels = kmeans.labels_

maxx = np.argmax(clustered_articles_matrix,axis=1)

for articles in hybrid_recommend_indexes :

    print('Recommended-Articles :')

    print('\n')

    print(title[articles][:no_of_recommends])
