
import pandas as pd
import recommender_functions as rfnc


class ArticleRecommender:

    """
    A class for providing personalized article recommendations for users or popular recommendations for new users.

    Attributes:
        df (pd.DataFrame): The user-item interaction dataset.
        df_content (pd.DataFrame): The article content dataset.
        article_meta (pd.DataFrame): Metadata summary for articles.
        user_meta (pd.DataFrame): Metadata summary for users.
        user_item (pd.DataFrame): User-item interaction matrix.
        tdidf (array): TD-IDF vectorization output for article titles.

    Methods:
        make_recs(user_id=None, article_id=None, m=10):
            Generates personalized article recommendations for a user or provides popular recommendations for new users.

    Example:
        # Instantiate the ArticleRecommender class
        recommender = ArticleRecommender()

        # Generate recommendations for a specific user
        recommender.make_recs(user_id=1, m=5)

        # Generate recommendations for a specific article
        recommender.make_recs(article_id=123, m=10)
    """

    def __init__(self):
        
        # Load datasets
        
        self.df = pd.read_csv('data/user-item-interactions.csv')
        self.df_content = pd.read_csv('data/articles_community.csv')
        del self.df['Unnamed: 0']
        del self.df_content['Unnamed: 0']

        # Convert article IDs to integer
        self.df['article_id'] = self.df['article_id'].astype(int)

        # Drop duplicates
        self.df_content.drop_duplicates(subset='article_id', inplace=True)

        # Encode email
        email_encoded = rfnc.email_mapper(self.df)
        del self.df['email']
        self.df['user_id'] = email_encoded

        # Create meta data summaries
        self.article_meta, self.user_meta = rfnc.create_meta_summary(self.df, self.df_content)

        # Construct user-item interaction matrix
        self.user_item = rfnc.create_user_item_interaction(self.df)

        # Construct TF-IDF matrix
        self.tdidf = rfnc.create_tfidf(self.article_meta, 'article_title')


    def make_recs(self, user_id=None, article_id=None, m=10):

        """
        Generate personalized article recommendations for a user or provide popular recommendations for new users.

        Args:
            user_id (int): User ID for whom recommendations are generated (optional).
            article_id (int): Article ID for which similar articles are recommended (optional).
            m (int): Number of recommendations to provide (optional).

        Returns:
            None: Prints personalized recommendations or popular articles.
        """
        
        print('Welcome to the IBM Data Science Platform')
        print('==========================================================================')
        print()
        
        if article_id is not None:
            article_name = self.article_meta[self.article_meta['article_id'] == article_id]['article_title'].iloc[0]
            recs_ = rfnc.get_content_recs(user_id, self.tdidf, self.user_item, self.user_meta, self.article_meta, article_id)
            content_recs = pd.merge(recs_, self.article_meta, on='article_id')
            content_recs.sort_values(by=['similarity_score', 'rank'], inplace=True)
            
            if len(content_recs) > 0:
                ctr = 0
                print(f"Recommendations for Article {article_id}: {article_name.title()}")
                print()
                for index, row in content_recs.iterrows():
                    print(f"  {row['article_id']}: {row['article_title'].title()} {row['star']}")
                    ctr += 1
                    if ctr > m:
                        break
                
                print()
                print('Note: articles with high user engagement are marked with asterisks')
            else:
                print('No similar titles found...')
            
            print()
            print('==========================================================================')
            return

        user_list = self.user_meta['user_id'].unique().tolist()

        if user_id not in user_list:
            pop_recs = rfnc.get_popular_recs(self.article_meta, m=10)
            print('If you are new, we recommend these popular articles:')
            print()
            
            for index, row in pop_recs.sort_values(by=['rank'], ascending=True).iterrows():
                if row['rank'] <= m:
                    print(f"  {row['article_id']}: {row['article_title'].title()} viewed by {int(row['unique_user_views'])}")
            
            print()
            print('==========================================================================')
            return
        
        elif self.user_meta.loc[self.user_meta['user_id'] == user_id, 'narticles'].iloc[0] < 6:
            user_narticles = self.user_meta.loc[self.user_meta['user_id'] == user_id, 'narticles'].iloc[0]
            print(f"You have viewed {user_narticles} article(s)")
            
            recs_ = rfnc.get_content_recs(user_id, self.tdidf, self.user_item, self.user_meta, self.article_meta)
            content_recs = pd.merge(recs_, self.article_meta, on='article_id')
            content_recs.sort_values(by=['similarity_score', 'rank'], inplace=True)
            
            content_recs_viewed = content_recs[content_recs['viewed'] == 1]
            content_recs_unseen = content_recs[content_recs['viewed'] == 0]

            if len(content_recs_viewed) > 0:
                ctr = 0
                print('Check out related articles:')
                print()
                
                for index, row in content_recs_viewed.iterrows():
                    print(f"  {row['article_id']}: {row['article_title'].title()} {row['star']}")
                    ctr += 1
                    if ctr > m:
                        break
            
            print()
            
            if len(content_recs_unseen) > 0:
                ctr = 0
                print('Explore related articles not viewed yet:')
                print()
                
                for index, row in content_recs_unseen.iterrows():
                    print(f"  {row['article_id']}: {row['article_title'].title()}")
                    ctr += 1
                    if ctr > m - 5:
                        break
            
            print()
            print('Note: articles with high user engagement are marked with asterisks')
            print()
            print('==========================================================================')
        
        else:
            user_narticles = self.user_meta.loc[self.user_meta['user_id'] == user_id, 'narticles'].iloc[0]
            print(f"You have viewed {user_narticles} article(s)")
            
            urecs_ = rfnc.get_user_recs(user_id, self.user_item, self.user_meta, m=5)
            user_recs = self.article_meta[self.article_meta['article_id'].isin(urecs_)]
            user_recs = user_recs.copy()
            user_recs.sort_values(by=['rank'], ascending=True, inplace=True)    

            if len(user_recs) > 0:
                ctr = 0
                print('Check out articles recommended by other users:')
                print()
                
                for index, row in user_recs.iterrows():
                    print(f"  {row['article_id']}: {row['article_title'].title()} {row['star']}")
                    ctr += 1
                    if ctr > m:
                        break
            else:
                print('No user recommendations at this time')
            
            crecs_ = rfnc.get_content_recs(user_id, self.tdidf, self.user_item, self.user_meta, self.article_meta)
            content_recs = crecs_.copy()
            content_recs = content_recs[~content_recs['article_id'].isin(user_recs['article_id'][:m])]
            content_recs = pd.merge(content_recs, self.article_meta, on='article_id', how='inner')
            content_recs.sort_values(by=['similarity_score', 'rank'], inplace=True)
            
            print()       
            if len(content_recs) > 0:                
                ctr = 0
                
                if len(user_recs) == 0:
                    print('Explore related articles:')
                else:
                    print('Explore related articles:')
                
                print()
                
                for index, row in content_recs.iterrows():            
                    print(f"  {row['article_id']}: {row['article_title'].title()} {row['star']}")
                    ctr += 1
                    
                    if ctr > m - 5:
                        break        

            print()
            print('Note: articles with high user engagement are marked with asterisks')
            print()
            print('==========================================================================')

        return