import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Loading data
# Contains information about the articles shared in the platform. Each article has its sharing date (timestamp), 
# the original url, title, content in plain text, the article' lang (Portuguese: pt or English: en) and information
# about the user who shared the article (author).
articles_df=pd.read_csv("shared_articles.csv")
articles_df=articles_df[articles_df['eventType']=='CONTENT SHARED']

# Loading the user interaction data for event type: 1. View 2. Like 3. Comment Created: 4. Follow 5. Bookmark
interactions_df=pd.read_csv("users_interactions.csv")

# Different interactions types, associated with a weight or strength
event_type_strength={'VIEW':1.0, 'LIKE':2.0, 'BOOKMARK':2.5, 'FOLLOW':3.0, 'COMMENT CREATED':4.0}
interactions_df['eventStrength']=interactions_df['eventType'].apply(lambda x:event_type_strength[x])

# Slicing the data to get an aggregate of users' whose interactions occur more than 5 times
users_interactions_count_df=interactions_df.groupby(["personId","contentId"]).size().groupby('personId').size()
# Further slice the data to get user interactions with more than 5
users_with_enough_interactions_df=users_interactions_count_df[users_interactions_count_df>=5].reset_index()[["personId"]]

# Doing a right join to only look at the users that have enough interactions
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
                                                            how = 'right', left_on = 'personId', right_on = 'personId')

# The interest level is a weighted sum of interaction type strength and apply a log transformation to smooth the distribution.
# Smoothed user preferences to get a smooth distribution esp on the column of event strength where first aggregation sum is 
# created and then smoothed
def smooth_user_preference(x):
    return math.log(1+x,2)

interactions_full_df = interactions_from_selected_users_df.groupby(["personId",'contentId'])['eventStrength'].sum() \
                                                                    .apply(smooth_user_preference).reset_index()

# Here a simple cross validation approach is used where 20% holdout is created.
interactions_train_df,interactions_test_df=train_test_split(interactions_full_df, 
                                                            stratify=interactions_full_df['personId'],
                                                            test_size=0.2,random_state=42)

''' Following section focuses on content based approach '''

#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()

# To model the user profile, we take all the item profiles the user has interacted and average them. The average is weighted
# by the interaction strength, in other words, the articles the user has interacted the most (eg. liked or commented) will 
# have a higher strength in the final user profile.
def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['contentId'])
    
    user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
    #Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed_df = interactions_full_df[interactions_full_df['contentId'].isin(articles_df['contentId'])].set_index('personId')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles

user_profiles = build_users_profiles()

# Actual implementation of the content based recommender 
class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', left_on = 'contentId',
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(articles_df)

''' Following section focuses on collaborative filtering approach '''

# using matrix factorization

#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', columns='contentId', values='eventStrength').fillna(0)
users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()

users_ids = list(users_items_pivot_matrix_df.index)

#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()

# Collaborative filtering
class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index() \
                                                        .rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                                                        .sort_values('recStrength', ascending = False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId',
                                                                                  'title', 'url', 'lang']]

        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, articles_df)

''' Following section focusses on hybrid filtering approach which combines the above two approaches '''

class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, items_df):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        #Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        #Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        #Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'inner', 
                                   left_on = 'contentId', 
                                   right_on = 'contentId')
        
        #Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrengthHybrid', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
    
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df)

# giving the final recommendation based on hybrid approach
print(hybrid_recommender_model.recommend_items(-1479311724257856983, topn=10, verbose=True))