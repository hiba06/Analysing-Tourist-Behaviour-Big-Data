# Analysing Tourist Behaviour using Big Data Technology

[![Python Version](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Spark Version](https://img.shields.io/badge/pyspark-2.4.5-orange.svg)](https://spark.apache.org/docs/2.4.5/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-1.0.2-blue.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project leverages Big Data technologies and machine learning to analyze geotagged photos from Flickr (specifically the London tourist area) to identify patterns in tourist behavior, cluster similar interest areas, and recommend tourist destinations based on users' interests.

---

## 📌 Project Overview

Understanding tourist behavior is crucial for urban planning, destination marketing, and personalized tour recommendations. However, analyzing massive datasets of geotagged tourist activity requires robust big data frameworks. 

This project implements:
1. **Big Data Ingestion:** Uses the Apache Spark (PySpark) framework to process large volumes of geotagged image metadata.
2. **Natural Language Processing (NLP):** Cleans and processes photo titles, descriptions, and tags using tokenization, stop-word removal, stemming (Porter Stemmer), and lemmatization (WordNet Lemmatizer).
3. **Feature Engineering:** Vectorizes text features using TF-IDF (Term Frequency-Inverse Document Frequency) representation.
4. **Constrained Clustering:** Groups similar interest destinations into balanced clusters using `KMeansConstrained` to ensure even cluster distributions.
5. **Smart Recommendation System:** Matches a user's typed interest keywords with the target cluster and recommends the top 5 tourist locations sorted by popularity (favourites/views).
6. **Model Explainability:** Integrates SHAP (Shapley Additive exPlanations) with a Random Forest Classifier to explain which keywords drive specific cluster classifications.

---

## 🛠️ Tech Stack & Dependencies

The project is built using:
- **Core Language:** Python 3.7.0
- **Big Data Engine:** Apache Spark (PySpark 2.4.5) for high-performance dataset loading and transformation
- **Machine Learning & NLP:** 
  - `scikit-learn` (TF-IDF Vectorization, MinMaxScaler, Random Forest)
  - `k-means-constrained` (Balanced KMeans clustering)
  - `nltk` (Natural Language Toolkit for stopwords, stemmers, and lemmatizers)
  - `shap` (SHAP model explanation framework)
- **Data Analysis & Viz:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Development Environment:** Jupyter Notebook

---

## 📁 Repository Structure

```text
├── Dataset/
│   └── tourism.csv              # Flickr geotagged dataset (photo_id, title, description, tags, faves)
├── model/
│   ├── tfidf.txt                # Saved TF-IDF Vectorizer (pickle)
│   └── tfidf_X.txt.npy          # Precomputed TF-IDF feature matrix (numpy)
├── .gitignore                   # Files and directories ignored by Git (caches, log files, checkpoints)
├── download_nltk.bat            # Windows batch script to trigger NLTK downloads
├── nltkdownload.py              # Python script containing NLTK download logic
├── requirements.txt             # Standard pip dependency list (for Python 3.7)
├── requirements_py311.txt       # Updated pip dependency list (for Python 3.11)
├── run.bat                      # Windows batch script to launch Jupyter Notebook
├── Tourist.ipynb                # Main Jupyter Notebook containing the code and analysis
└── README.md                    # Project documentation (this file)
```

---

## 💾 Dataset Details

The project utilizes geotagged photo metadata from Flickr in London.
- **Source:** [Kaggle Flickr London Dataset](https://www.kaggle.com/datasets/amiralisa/flickr_london/data)
- **Primary Schema:**
  - `photo_id`: Unique identifier of the geotagged image
  - `title`: User-defined title of the photo
  - `description`: User-provided description
  - `tags`: Metadata tags associated with the photo (e.g. landmarks, attractions)
  - `faves`: Number of times the photo was favorited (proxy for tourist traffic and popularity)

---

## 🚀 Setup & Installation

Follow these steps to run the project on your local machine:

### 1. Prerequisites
- Install **Python 3.7.0** ([Download Python 3.7.0](https://www.python.org/downloads/release/python-370/)).
- Ensure Java 8 is installed and the `JAVA_HOME` environment variable is configured (required for Apache Spark / PySpark).

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/Analysing-Tourist-Behaviour.git
cd Analysing-Tourist-Behaviour
```

### 3. Install Dependencies
Open your command prompt/terminal and run the appropriate command for your Python version:

* **For Python 3.7 (Original):**
  ```cmd
  pip install -r requirements.txt
  ```

* **For Python 3.11+ (Modern installations):**
  ```cmd
  pip install -r requirements_py311.txt
  ```

### 4. Download NLTK Datasets
NLTK requires downloading lexical resources (stopwords and lemmatizers) before execution. Run the batch script:
```bash
# On Windows, double-click download_nltk.bat or run:
download_nltk.bat

# Or run manually via Python:
python nltkdownload.py
```
This script downloads `stopwords`, `wordnet`, and `omw-1.4` programmatically.

---

## 💻 How to Run

1. Launch Jupyter Notebook by double-clicking `run.bat` or by running:
   ```bash
   jupyter notebook
   ```
2. In the Jupyter interface, open **`Tourist.ipynb`**.
3. Execute the cells sequentially (`Shift + Enter`).
4. **Interactive Recommendations:**
   When you reach the recommendation cell, it will prompt you:
   ```text
   Enter your interest: 
   ```
   Provide keywords (e.g. `britishmuseum london sculpture roman` or `buckinghampalace themall` or `hydepark`) to fetch top recommendations matching your interest profile.

---

## 📊 Visualizations and Outputs

The notebook generates multiple visual plots:

### 1. Cluster Visualization
Plots the generated clusters of tourist locations against the volume of visits (`faves`), illustrating groups of users with similar interests:
* **X-Axis:** Tourist location tags / landmarks (grouped by cluster similarity)
* **Y-Axis:** Number of visits / favorites

### 2. SHAP Explainability Plot
Provides a summary plot showing feature importances for clustering. The random forest classifier is trained on a subset of the TF-IDF matrix to predict the K-Means cluster labels. The SHAP summary plot highlights which words (e.g., specific landmark names like `museum`, `park`, `palace`) have the highest predictive power for determining each cluster.

---

## 📝 License

This project is licensed under the MIT License. Feel free to use and modify it for educational or research purposes.
