
#-------------------------------------------------------------------------------------------------------------#
#  Functions used in the recommender class                                                                    #
#-------------------------------------------------------------------------------------------------------------#


import pandas as pd
import numpy as np
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def email_mapper(df):
    ''' 
    DESCRIPTION:
        Maps email addresses to unique integer values.

    INPUT:
        - df (DataFrame): The DataFrame containing an 'email' column with email addresses.

    OUTPUT:
        - email_encoded (list): A list of integers representing the encoded email addresses.
    
    Example:
        ```
        email_encoded = email_mapper(df)
        ```
    '''

    coded_dict = {}
    cter = 1
    email_encoded = []

    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1
        email_encoded.append(coded_dict[val])

    return email_encoded

def create_meta_summary(df, df_content):
    ''' 
    DESCRIPTION:
        Creates a meta summary at the article level and user level.

    INPUT:
        - df (DataFrame): A DataFrame with 'article_id', 'title', and 'user_id' columns.
        - df_content (DataFrame): A DataFrame with 'article_id' and 'doc_full_name' columns.

    OUTPUT:    
        - art_ (DataFrame): Article aggregated summary (viewed and unseen).
        - user_ (DataFrame): User aggregated summary.
    
    Example:
        ```
        article_meta, user_meta = create_meta_summary(df, df_content)
        ```
    '''

    # Calculate user interactions by article
    art_ = df.groupby(['article_id', 'title']).agg({'user_id': ['nunique', 'count']}).reset_index(drop=False)
    art_.columns =  ['article_id', 'title', 'unique_user_views', 'tot_user_views']    

    # Create a DataFrame of all articles, viewed and not viewed
    art_ = pd.merge(df_content[['article_id', 'doc_full_name']], art_, on='article_id', how='outer')    
    art_['viewed'] = np.where(art_['unique_user_views'].isnull(), 0, 1)
    art_.fillna(0, inplace=True)
    art_['article_title'] = np.where(art_['doc_full_name'] != 0, art_['doc_full_name'], art_['title'])
    art_['rank'] = art_['unique_user_views'].rank(ascending=False, method='dense').astype(int)
    art_['star'] = np.where(art_['rank'] < 11, '*****',
                    np.where(art_['rank'] < 26, '****',
                    np.where(art_['rank'] < 51, '***', '')
                    ))       
    art_.sort_values(by=['rank'], ascending=True, inplace=True)    

    # Calculate article views by users
    user_ = df.groupby('user_id')['article_id'].nunique().reset_index()
    user_.columns = ['user_id', 'narticles']
    user_.sort_values(by=['user_id'], ascending=True, inplace=True)

    return art_, user_

def create_user_item_interaction(df):
    '''
    DESCRIPTION:
        Creates a user x article interaction matrix with user IDs as rows and 
        article IDs as columns, with 1 values where a user interacted with an article and 0 otherwise.

    INPUT:
        - df (DataFrame): A DataFrame with 'article_id', 'title', and 'user_id' columns.

    OUTPUT:
        - user_item (DataFrame): User-item interaction matrix.
    
    Example:
        ```
        user_item = create_user_item_interaction(df)
        ```
    '''

    user_item_ = df.groupby(['user_id', 'article_id'])['user_id'].nunique().unstack()
    user_item_.fillna(0, inplace=True)
    
    return user_item_ 

def customtokenize(text):
    '''
    DESCRIPTION:
        Performs custom text tokenization by normalizing text, 
        removing specified patterns, removing stopwords, tokenizing, and lemmatizing the text.

    INPUT:
        - text (str): Input text to be tokenized.

    OUTPUT:
        - lemma_ (list): Tokenized and lemmatized words.
    '''
    
    # Normalize text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    text = re.sub(r'(?:\b\d+\b)', ' ', text)    
               
    # Stopword list 
    stop_words = stopwords.words("english")
        
    # Tokenize
    words = word_tokenize(text)
        
    # Lemmatize
    lemmed_ = [WordNetLemmatizer().lemmatize(w).strip() for w in words if w not in stop_words]

    return lemmed_    

def create_tfidf(df, title):
    '''
    DESCRIPTION:
        Constructs a TF-IDF matrix.

    INPUT:
        - df (DataFrame): A DataFrame containing the title.
        - title (str): The column in df that contains the title. 

    OUTPUT:
        - matrix (np.ndarray): TD-IDF vectorization matrix.
      
    Example:
        ```
        tfidf_matrix = create_tfidf(article_meta, 'article_title')
        ```
    '''

    # Create a corpus of titles
    corpus = list(df[title].unique())

    # Run TD-IDF vectorization
    vectorizer = TfidfVectorizer(tokenizer=customtokenize, ngram_range=(1, 3), min_df=3, use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return tfidf_matrix

def get_article_meta(article_id, article_meta):
    ''' 
    DESCRIPTION:
        Returns article metadata for a given article ID.

    INPUT:
        - article_id (int): The article ID.
        - article_meta (DataFrame): The DataFrame containing article metadata.

    OUTPUT:
        - article (Series): Article metadata as a Series.
    
    Example:
        ```
        article_info = get_article_meta(article_id, article_meta)
        ```
    '''

    if article_id not in article_meta['article_id'].unique().tolist():
        print(f"Error: Article {article_id} is not in the dataset.")
        return    
    
    else:
        return article_meta[article_meta['article_id'] == article_id]

    
def find_user_articles(user_id, user_item, user_meta):
    '''
    DESCRIPTION:
        Provides a list of article IDs that have been seen by a user.

    INPUT:
        - user_id (int): User ID.
        - user_item (DataFrame): User-item interaction matrix where 1's indicate user-article interactions.
        - user_meta (DataFrame): User metadata.

    OUTPUT:
        - article_ids (list): List of article IDs seen by the user.
    
    Example:
        ```
        seen_articles = find_user_articles(user_id, user_item, user_meta)
        ```
    '''

    if user_id not in user_meta['user_id'].unique().tolist():
        print(f"Error: User {user_id} is not in the dataset. Valid user IDs are between 1 and {user_meta['user_id'].max()}.")
        return  
        
    # Fetch article IDs where the user interaction is 1 (has interacted)
    article_ids = user_item.loc[user_id, user_item.loc[user_id] == 1].index.tolist()

    return article_ids


def get_similar_articles(article_id, tfidf, article_meta, sim_threshold=0.1, m=50):
    '''
    DESCRIPTION:
        Find articles with titles most similar to the input article ID.

    INPUT:
        - article_id (int): Article ID to search for similarities with.
        - tfidf (array): TF-IDF vectorization output.
        - article_meta (DataFrame): Article metadata.
        - sim_threshold (float): Minimum similarity score to consider.
        - m (int): Number of similar articles to return.

    OUTPUT:
        - similar_content (DataFrame): DataFrame of similar articles sorted by similarity score.
    
    Example:
        ```
        similar_articles = get_similar_articles(article_id, tfidf, article_meta)
        ```
    '''

    if article_id not in article_meta['article_id'].unique().tolist():
        print(f"Error: Article {article_id} is not in the dataset.")
        return    
    
    # Compute cosine similarity
    articles_similarity = cosine_similarity(tfidf, tfidf)
    sim_df = pd.DataFrame(articles_similarity, columns=article_meta['article_id'].tolist(), index=article_meta['article_id'].tolist())    

    # Retrieve the row corresponding to the search ID but exclude the search ID column    
    sim_filter = sim_df.loc[article_id].drop(article_id, errors='ignore')
    sim_filter.sort_values(ascending=False, inplace=True)

    # Filter values greater than the threshold
    sim_filter = sim_filter[sim_filter > sim_threshold]

    similar_content = pd.DataFrame({'article_id': sim_filter.index, 'similarity_score': sim_filter.values})

    if len(similar_content) > 1:    
        similar_content = similar_content.sort_values(by=['similarity_score'], ascending=False)
        similar_content = similar_content[:m]

    return similar_content


def get_content_recs(user_id, tfidf, user_item, user_meta, article_meta, article_id=None, m=10):
    '''
    DESCRIPTION:
        Retrieve content-based recommendations for a user.

    INPUT:
        - user_id (int): User ID.
        - tfidf (array): TF-IDF vectorization output.
        - user_item (DataFrame): User x item interaction matrix.
        - user_meta (DataFrame): User metadata.
        - article_meta (DataFrame): Article metadata.
        - article_id (int, optional): Article ID for which recommendations are requested.
        - m (int): Number of similar articles to return for each article viewed.

    OUTPUT:
        - content_recs (DataFrame): Ordered article recommendations.

    Example:
        ```
        content_recommendations = get_content_recs(user_id, tfidf, user_item, user_meta, article_meta)
        ```

    '''

    if article_id is not None:
        content_recs = get_similar_articles(article_id, tfidf, article_meta)
        content_recs['ref_id'] = article_id
        content_recs.sort_values(by=['similarity_score'], ascending=False, inplace=True)
        return content_recs

    if user_id not in user_meta['user_id'].unique().tolist():
        print(f"Error: User {user_id} is not in the dataset. Valid user IDs are between 1 and {user_meta['user_id'].max()}.")
        return

    # Get previously viewed articles
    seen_articles = find_user_articles(user_id, user_item, user_meta)
    
    content_recs = pd.DataFrame()

    # Find similar articles based on viewed articles
    for id in seen_articles:                 
        if content_recs.empty:
            content_recs = get_similar_articles(id, tfidf, article_meta)
            content_recs['ref_id'] = id            
        else:
            new_recs = get_similar_articles(id, tfidf, article_meta)
            new_recs['ref_id'] = id
            content_recs = pd.concat([content_recs, new_recs])            
            # Filter out any seen articles
            content_recs = content_recs[~content_recs['article_id'].isin(seen_articles)]

    def top_records(group, n=1):
        return group.head(n)

    if len(content_recs) > 1:
        content_recs.sort_values(by=['ref_id', 'similarity_score'], ascending=False)                        
        content_recs.drop_duplicates(inplace=True)    
        # Apply the custom function to get the top records for each ref_id
        content_recs = content_recs.groupby('ref_id', group_keys=False).apply(top_records, n=m)    

    content_recs.sort_values(by=['similarity_score'], ascending=False, inplace=True)    

    return content_recs


def get_popular_recs(article_meta, m=25):
    '''
    DESCRIPTION:
        Returns popular recommendations based on weighted user views.

    INPUT:
        - article_meta (DataFrame): Article metadata.
        - m (int): Maximum number of recommended articles to display.

    OUTPUT:
        - pop_recs (DataFrame): Ordered popular recommendations.

    Example:
        ```
        popular_recommendations = get_popular_recs(article_meta)
        ```

    '''

    pop_recs = article_meta.loc[article_meta['rank'] <= m, ['article_id', 'article_title', 'rank','unique_user_views']]
    pop_recs.sort_values(by=['rank'], ascending=True, inplace=True)
    
    return pop_recs


def get_similar_users(user_id, user_item, user_meta, sim_threshold=3, cnt_threshold=5):
    '''
    DESCRIPTION:
        Finds similar users and returns only those with a high number of article interactions.

    INPUT:
        - user_id (int): User ID.
        - user_item (DataFrame): User x item interaction matrix (1's when a user has interacted with an article, 0 otherwise).
        - user_meta (DataFrame): User metadata.
        - sim_threshold (int): Threshold limit to include similar users.
        - cnt_threshold (int): Threshold limit of article interactions.

    OUTPUT:
        - similar_users (DataFrame): Ordered similar users.

    Example:
        ```
        similar_users = get_similar_users(0, user_item, user_meta)
        ```
    '''

    if user_id not in user_meta['user_id'].unique().tolist():
        print(f"Error: User {user_id} is not in the dataset. Valid user IDs are between 1 and {user_meta['user_id'].max()}.")
        return

    # Compute similarity of each user to the provided user
    user_similarity = np.dot(user_item.loc[user_id], user_item.T)

    # Sort similarities in descending order and get the corresponding user IDs
    # Exclude the similarity with itself
    sorted_ids = np.argsort(user_similarity)[::-1]
    
    similar_users = pd.DataFrame({'user_id': user_item.index[sorted_ids][1:], 
                                  'similarity_score': user_similarity[sorted_ids][1:]})
    
    similar_users = similar_users[similar_users.similarity_score > sim_threshold]
    similar_users = pd.merge(similar_users, user_meta, on='user_id')
    similar_users = similar_users[similar_users['narticles'] > cnt_threshold]

    if len(similar_users) > 1:
        similar_users.sort_values(by=['similarity_score', 'narticles'], ascending=[False, False], inplace=True)

    similar_users = similar_users[['user_id', 'similarity_score', 'narticles']]

    return similar_users

def get_user_recs(user_id, user_item, user_meta, m=5):
    '''
    DESCRIPTION:
        Loops through the users based on closeness to the input user_id.
        For each user, finds articles the user hasn't seen before and provides them as recommendations.

    INPUT:
        - user_id (int): User ID.
        - user_item (DataFrame): User x item interaction matrix (1's when a user has interacted with an article, 0 otherwise).
        - user_meta (DataFrame): User metadata.
        - m (int): Number of users with the most article interactions to consider.

    OUTPUT:
        - user_recs (list): Unordered recommendations for the user.
    '''

    if user_id not in user_meta['user_id'].unique().tolist():
        print(f"Error: User {user_id} is not in the dataset. Valid user IDs are between 1 and {user_meta['user_id'].max()}.")
        return

    articles_seen = find_user_articles(user_id, user_item, user_meta)
    similar_users = get_similar_users(user_id, user_item, user_meta)

    user_recs = set()

    if len(similar_users) > 0:
        # Take recommendations from top similar users
        for user in similar_users['user_id'][:m].unique():
            if len(user_recs) == 0:
                recs = find_user_articles(user, user_item, user_meta)
                user_recs = set(recs)
            else:
                recs = find_user_articles(user, user_item, user_meta)
                user_recs.update(recs)

        user_recs = list(user_recs - set(articles_seen))
    else:
        user_recs = list(user_recs)

    return user_recs


def print_user_articles(user_id, user_item, user_meta, article_meta):
    '''
    DESCRIPTION:
        Prints articles viewed by a user and their titles.

    INPUT:
        - user_id (int): User ID.
        - user_item (DataFrame): User x item interaction matrix (1's when a user has interacted with an article, 0 otherwise).
        - user_meta (DataFrame): User metadata.
        - article_meta (DataFrame): Article metadata.

    OUTPUT:
        None (Prints the articles viewed by the user and their titles).

    '''
    if user_id is not None:
        ids = find_user_articles(user_id, user_item, user_meta)
        print(f"User {user_id} viewed {len(ids)} articles:")
        
        top_25 = article_meta.loc[article_meta['rank'] < 25, 'article_title'].unique()
        article_names = set()
        
        for id in ids:
            article = get_article_meta(id, article_meta)
            article_names.update(article['article_title'])
            
        print([title.title() for title in article_names if title in top_25])


