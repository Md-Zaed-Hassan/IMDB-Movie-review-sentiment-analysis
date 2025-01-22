# Sentiment Analysis of IMDB Movie Reviews

This project focuses on sentiment analysis of IMDB movie reviews, leveraging various machine learning models to classify reviews as either **positive** or **negative**. The project involves data preprocessing, visualization, and model comparison using metrics like accuracy, precision, and recall.

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Workflow](#workflow)
- [Models](#models)
- [Results](#results)
- [Visualization](#visualization)
- [How to Run](#how-to-run)

## Dataset
The dataset consists of 50,000 IMDB reviews labeled as positive or negative. Reviews are preprocessed to remove HTML tags, stopwords, and other unnecessary elements.

- Source: [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

## Requirements
The following libraries are required to run this project:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- WordCloud

## Workflow
1. **Data Loading**: Load the dataset from Google Drive.
2. **Data Cleaning**: 
   - Remove duplicates and null values.
   - Preprocess reviews (lowercase conversion, removal of stopwords, stemming).
3. **Feature Engineering**:
   - Create `review_length` as a new feature.
   - Transform text data using `TfidfVectorizer`.
4. **Visualization**:
   - Generate word clouds for positive and negative reviews.
   - Plot histograms, scatter plots, and heatmaps.
5. **Model Training and Evaluation**:
   - Models: Random Forest, Gradient Boosting, Logistic Regression, SVM, Decision Tree.
   - Metrics: Accuracy, F1-score, Precision, Recall.

## Models
The following models were implemented and evaluated:
- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree

## Results
- **Best Accuracy**: Logistic Regression with 89%.
- **Other Models**: 
  - Random Forest: 86%
  - Gradient Boosting: 85%
  - SVM: 89%
  - Decision Tree: 72%

## Visualization
Key visualizations include:
- Word clouds for sentiment-based analysis.
- Model accuracy comparison using bar plots.
- Heatmaps showing feature correlations and F1 scores.

## How to Run
1. Clone the repository or download the notebook file.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
