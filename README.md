# Sentiment Analysis on Twitter and Reddit

### Monitoring Public Sentiment on Brands Using Social Media Data for Strategic Insights  
**Contributors:** Parnavi Sen, Navneet Parab  
**Date:** April 2025  

---

## Project Overview

Public sentiment significantly impacts a brand’s reputation, market value, and customer trust. Yet, many brands lack the tools to analyze real-time feedback shared by users on platforms like Twitter and Reddit. This project presents a robust NLP pipeline to monitor, classify, and visualize sentiment trends for brands using social media data.

We leverage both traditional models and state-of-the-art transformer-based deep learning architectures to understand nuanced, context-aware sentiment and provide actionable insights to brands for product and communication strategy.

---

## Project Goals

- Collect and preprocess large-scale datasets from Twitter and Reddit.
- Apply lexicon-based, clustering, and transformer models for sentiment classification.
- Compare model performance (accuracy, interpretability, efficiency).
- Use LLMs (BERT, Gemini) to:
  - Detect mixed or ambiguous sentiment.
  - Extract contextual sentiment cues.
  - Summarize trends and topic clusters.
- Visualize sentiment trends over time with interactive dashboards.
- Provide actionable insights to enhance brand strategy and engagement.

---

## Dataset Details

The Reddit dataset was collected using the PRAW (Python Reddit API Wrapper) and includes:

- `Subreddit`: Community where the post originated.
- `Timestamp`: Post creation date and time.
- `Title` and `Body`: Main textual content.
- `First Comment`: Top user comment on the post.
- `Upvotes` and `Upvote Ratio`: Measures of community approval.
- `Number of Comments`: Engagement metric.

Two domains were analyzed:
- Athletic Apparel Brands
- Technology Companies

---

## Preprocessing Pipeline

### Text Clean-Up
- Lowercased text
- Removed punctuation, special characters, numbers
- Removed stopwords (using NLTK)

### NLP Tasks
- Tokenization  
- Lemmatization  
- POS tagging  
- Emoji normalization (using `emoji` package)

### Transformer Input Processing
- BERT subword tokenization
- Attention masks and padding
- Uniform input formatting for deep models

---

## Models Used

### Lexicon-Based
- VADER: Lightweight rule-based model for polarity scoring

### Unsupervised Learning
- K-Means: Clustered sentiment-rich post vectors (TF-IDF) to create pseudo-labels

### Transformer-Based Deep Learning
- BERT (`bert-base-uncased`)
  - Context-aware embedding
  - Fine-tuned on pseudo-labeled Reddit data
  - Achieved 85–90% accuracy

- Gemini
  - Efficient transformer architecture (used in place of LLaMA)
  - Fine-tuned with mixed precision & gradient checkpointing
  - Lower memory footprint, good for deployment

---

## Results

| Model   | Type       | Accuracy | Notes |
|---------|------------|----------|-------|
| VADER   | Rule-based | –        | Used for pseudo-labels |
| K-Means | Clustering | –        | Unsupervised label generation |
| BERT    | Transformer| 85–90%   | Best performance, handles context and sarcasm |
| Gemini  | Transformer| ~80–85%  | Lightweight alternative to BERT |

- Visualizations: Sentiment trends, confusion matrices, accuracy/loss curves.
- Findings:
  - Neutral sentiment dominated across both domains.
  - Positive sentiment reflected strong brand loyalty.
  - Sarcasm and ambiguity remained challenges.

---

## Tools and Technologies

### Languages and Frameworks
- Python  
- TensorFlow, PyTorch  
- Scikit-learn, Transformers (Hugging Face)

### Libraries
- Data Collection: `PRAW`, `snscrape`
- NLP: `nltk`, `spaCy`, `emoji`
- Modeling: `transformers`, `scikit-learn`, `tensorflow`
- Visualization: `matplotlib`, `seaborn`, `plotly`


---

## Expected Output

- Trained sentiment classification model (BERT and Gemini variants)
- Visualization dashboards for:
  - Sentiment over time
  - Top keywords per sentiment class
  - Emerging discussion themes
- Strategic insights to help brands respond to public opinion

---

## Future Work

- Improve sarcasm detection using models like RoBERTa, T5
- Expand to additional platforms (Twitter, Instagram)
- Add domain-specific fine-tuning for healthcare, finance, etc.
- Use active learning or semi-supervised methods to improve label quality
- Develop lightweight APIs and deploy on edge devices

---

## Contact

For questions, collaboration, or feedback:  
Parnavi Sen — [parnavi.sen.ps@gmail.com](mailto:parnavi.sen.ps@gmail.com)  
Navneet Parab — [navneetparabb20@gmail.com](mailto:navneetparabb20@gmail.com)

---


