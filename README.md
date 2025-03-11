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
## Dataset Details
The dataset consists of Reddit posts scraped using PRAW (Python Reddit API Wrapper). It includes real-time Reddit data with the following variables:

- Subreddit: The community where the post was published.
- Timestamp: The time when the post was created.
- Title:  The title of the Reddit post.
- Body: The full text content of the post.
- Comments: The first comment on the post.
- Number of Upvotes: The number of upvotes received.
- Upvote Ratio: The ratio of upvotes to downvotes.
### Preprocessing Steps
#### Text Clean-Up
- Standardize text to lowercase.
- Remove special characters and numbers.
- Removed stopwards
#### NLP Breakdown
- Tokenization: Splitting text into individual words or tokens.
- Lemmatization: Reducing words to their base or root form.
- Part of Speech (POS) Tagging: Identifying grammatical roles of words.

## Machine Learning Implementations
### Traditional Machine Learning Models
#### Vader Implementations
- Applied VADER, a rule-based lexicon approach for sentiment classification, especially effective for short social media texts.
- Useful for quick, lightweight sentiment classification before applying more complex deep learning models.
### K-Means Clustering
- Applied unsupervised learning to group similar posts based on sentiment and emerging discussion themes.
- Helps in detecting latent sentiment groups and customer concerns without prior labels.
### Deep Learning and Transformer-Based Models
## BERT (bert-base-uncased) Implementation
- A transformer-based model capable of capturing contextual sentiment by analyzing bidirectional word relationships.

- Tokenization is performed using BERT tokenizer, which splits words into subwords for better handling of out-of-vocabulary words.

- Padding is applied to ensure all sequences have uniform lengths for batch processing.

- Outperforms traditional models in understanding sentiment variations, sarcasm, and mixed opinions.
## LLaMA (Large Language Model Meta AI) Implementation
- A large-scale transformer-based model designed for nuanced sentiment understanding.

- Excels at processing longer social media discussions and identifying subtle sentiment cues.

- Particularly useful for domain-specific sentiment classification by fine-tuning on Reddit-based datasets.

- Enhances performance by capturing context-dependent sentiments that traditional models struggle with.
## Expected Output
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
