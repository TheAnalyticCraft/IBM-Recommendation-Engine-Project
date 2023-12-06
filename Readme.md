## Article Recommendations for IBM Watson Studio platform

### Overview

This Udacity graded project is part of the Experimental Design and Recommendations course, focusing on analyzing user interactions with articles on the IBM Watson Studio platform and constructing personalized article recommendations.

This project is structured into separate sections, each representing a step in the creation of a recommendation engine. It explores various approaches:

- Rank-Based Recommendations: This method suggests articles with the highest interaction rates (views).
- Collaborative Filtering: Recommendations are generated based on user similarity.
- Content-Based Recommendations: Recommendations rely on content similarity between articles.

A detailed walkthrough of the steps is described in the `Recommendations_with_IBM jupyter` notebook or HTML. Udacity provided an outline for which functions need to be written to perform the different tasks in this recommendation exercise, as shown in sections:

- II. Rank-based  
- III. User-User Collaborative - Filtering  
- IV. Content-Based Filtering  

At the end of section IV, a demonstration of three combined approaches is presented to construct a basic recommendation engine, addressing various user scenarios:

- New Users: Recommendations prioritize popular (most viewed) articles.  
- Less-Engaged Users (with 5 or fewer interactions): Content-based articles are recommended, sorted by popularity.  
- Active Users: Recommendations encompass both user-based and content-based articles, sorted by both similarity and popularity.  

To facilitate this process, a Python class named ArticleRecommender (contained in `recommender.py`) serves as a central hub for processing. It illustrates how different user scenarios are handled effectively. Additional functions from `recommender_functions.py` are imported to support these recommendations.

In Section V, we delve into Matrix Factorization, specifically Singular Value Decomposition (SVD), to predict user-article interactions. SVD offers advantages in handling data sparsity and addressing the cold start problem, making it well-suited for scenarios with binary interactions (0 or 1), representing implicit ratings.

However, it's crucial to assess the quality of these predictions, as they significantly influence the accuracy of the similarity matrix. During evaluation, as the number of latent factors increases, SVD performs exceptionally well on the training data, indicating a strong fit. However, when applied to the test dataset, SVD's performance deteriorates, raising concerns about potential overfitting issues.

Addressing overfitting in SVD predictions requires further exploration and refinement.

### About the Data

The dataset comprises 45,993 observations of user-item interactions, involving 714 unique articles and 5,148 users. The median number of times an article is viewed is 25, indicating moderate engagement. Notably, half of the users have viewed three articles or fewer.

(https://github.com/TheAnalyticCraft/IBM-Recommendation-Engine-Project/blob/main/user_item_visualizations.png)  

The article community file contains a total of 1,051 unique article titles. Among these, 227 out of the 714 viewed articles are not included in the article community file. This results in a combined total of 1,328 articles, including those that have been viewed and those that have not been seen by users.

In the absence of explicit ratings, the data is represented in a binary format, where 1 signifies that an article has been viewed, regardless of the frequency of views, and 0 indicates otherwise. It's important to note that a 0 does not necessarily imply disinterest, and multiple views can reflect various user behaviors, from simple clicks to a strong liking for the article.

However, this binary representation may not always capture the full spectrum of user preferences and interactions. Similarity calculations based solely on binary data may not always be indicative of strong or nuanced connections between articles, potentially leading to limitations in the recommendations.

### Documentation

**IBM Data**

- `/data/articles_community.csv`: Catalog of articles.
- `/data/user-item-interaction.csv`: User views of articles.

**File Description**

- `Recommendations_with_IBM.ipynb`: Jupyter notebook containing the Udacity exercises.
- `Recommendations_with_IBM.html`: HTML version of the Jupyter notebook.
- `recommender.py`: Builds the recommender class.
- `recommender_functions.py`: Contains functions for recommender class.
- `project_test.py`: Contains grading functions. 
- `top_5.p`, `top_10.p`, `user_item_matrix.p`: Udacity files.

**Dependencies**

- Python 3.6.3
- Main Python Packages:
  - Scikit-Learn
  - Pandas
  - NumPy
  - NLTK
  - Plotly

**Clone repository** 

```
git clone https://github.com/TheAnalyticCraft/IBM-Recommendation-Engine-Project.git
```

### Acknowledgements

* [Udacity](https://www.udacity.com/) for offering an exceptional Experimental Testing and Recommendation Engine.
* [IBM](https://dataplatform.cloud.ibm.com) for providing the dataset essential for building a recommendation engine.
