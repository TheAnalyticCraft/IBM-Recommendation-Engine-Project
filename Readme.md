## Recommender Engine for IBM Watson Studio platform

### Overview

This Udacity graded project is part of the Experimental and Recommendation course, focusing on analyzing user interactions with articles on the IBM Watson Studio platform and constructing personalized article recommendations.

This project is structured into separate sections, each representing a step in the creation of a recommendation engine. It explores various approaches:

- Rank-Based Recommendations: This method suggests articles with the highest interaction rates.
- Collaborative Filtering: Recommendations are generated based on user similarity.
- Content-Based Recommendations: Recommendations rely on content similarity between articles.

A detailed walkthrough of the steps is described in the notebook or HTML file. Udacity provided an outline for which functions need to be written to perform the different tasks in this recommendation exercise, as shown in sections:

- II. Rank-based  
- III. User-User Collaborative - Filtering  
- IV. Content-Based Filtering (optional)  

At the end of section IV, a demonstration of three combined approaches is presented to construct a basic recommendation system, addressing various user scenarios:

- New Users: Recommendations prioritize popular (most viewed) articles.  
- Less-Engaged Users (with 5 or fewer interactions): Content-based articles are recommended, sorted by popularity.  
- Active Users: Recommendations encompass both user-based and content-based articles, sorted by both similarity and popularity.  

To facilitate this process, a Python class named ArticleRecommender (found in recommender.py) serves as a central hub for processing. It illustrates how different user scenarios are handled effectively. Additional functions from "recommender_functions.py" are imported to support these recommendations.


### Documentation

**IBM Data**

- `/data/articles_community.csv`: Catalog of articles.
- `/data/user-item-interaction.csv`: User views of articles.

**File Descriptions**

- `Recommendations_with_IBM.ipynb`: Jupyter notebook containing the exercises.
- `Recommendations_with_IBM.html`: HTML version of the Jupyter notebook.
- `recommender.py`: Builds the recommender class.
- `recommender_functions.py`: Contains functions by recommender class.
- `project_test.py`: Contains grading functions. 
- `top_5.p`, `top_10.p`, `user_item_matrix.p`: Provided by Udacity.

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

* [Udacity](https://www.udacity.com/) for offering an exceptional Experimental Testing and Recommendation Engine
* [IBM](https://dataplatform.cloud.ibm.com) for providing the dataset essential for building a recommendation system
