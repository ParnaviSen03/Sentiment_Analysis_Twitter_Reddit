# Sentiment_Analysis_Twitter_Reddit 
## Monitoring Public Sentiment on Brands Using Social Media Data for Strategic Insights 
## Project Overview
Public sentiment significantly impacts a brand's reputation and business success. However, many brands lack the tools to monitor and analyze the massive volumes of real-time feedback available on social media. This project focuses on leveraging social media data from platforms like Twitter and Reddit to monitor public sentiment, identify trends, and provide strategic insights for brands to improve decision-making and respond proactively. 
## Key Features and Goals
### Project Goals
1. Collect, clean, and preprocess large datasets from Twitter and Reddit.
2. Build and evaluate a sentiment analysis model to classify public opinion into Positive, Neutral, or Negative categories.
3. Leverage Large Language Models (LLMs) to:
- Perform context-aware sentiment analysis.
- Identify nuanced opinions and mixed sentiments.
- Summarize sentiment trends and extract key themes.
4. Create insightful visualizations to track sentiment trends over time.
### Dataset Details
The dataset consists of Reddit posts scraped using PRAW (Python Reddit API Wrapper). It includes real-time Reddit data with the following variables:

- Subreddit: The community where the post was published.
- Timestamp: The time when the post was created.
- Title:  The title of the Reddit post.
- Body: The full text content of the post.
- Comments: The number of comments on the post.
- Number of Upvotes: The number of upvotes received.
-Upvote Ratio: The ratio of upvotes to downvotes.
#### Preprocessing Steps
##### Text Clean-Up
- Standardize text to lowercase.
- Remove special characters and numbers.
- Removed stopwards
##### NLP Breakdown
- Tokenization: Splitting text into individual words or tokens.
- Lemmatization: Reducing words to their base or root form.
- Part of Speech (POS) Tagging: Identifying grammatical roles of words.

### Expected Output
- Sentiment Analysis Model: A trained and validated model to classify text sentiment.
- Visualization Dashboards: Graphs and charts showcasing sentiment trends, keyword frequency, and emerging themes.
- Actionable Insights: Provide brands with tools to track public opinion and optimize marketing campaigns while addressing customer concerns.
## Tools and Technologies
### Programming Languages
- Python: Used for data collection, preprocessing, analysis, and model development.
### Libraries
- Data Collection: snscrape, PRAW.
- Preprocessing & Analysis: Pandas, NumPy, nltk, spaCy.
- Modeling: transformers, Scikit-learn, TensorFlow, PyTorch.
- Visualization: Matplotlib, Seaborn, Plotly.
- Deployment: AWS.
### Platforms
- Google Colab or Jupyter Notebook for experimentation.
- Tableau or Power BI for advanced visualizations.
- AWS for hosting the application.
## Contact
For questions, suggestions, or feedback, please reach out to:
- Parnavi: parnavi.sen.ps@gmail.com
- Navneet: navneetparabb20@gmail.com