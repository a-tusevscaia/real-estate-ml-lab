# Real Estate ML Lab

Six machine learning labs on one real estate dataset, covering the main unsupervised and supervised techniques: regression, classification, clustering, dimensionality reduction, pattern mining and anomaly detection.

Each notebook is a self-contained workflow: data loading and cleaning, exploratory analysis, modelling, evaluation and written conclusions with limitations.

> **Note:** the notebooks are written in Russian. Code, plots and metric names are standard, so the results can be followed without reading the text.

## Dataset

A synthetic real estate dataset: **2,840 listings, 28 features** (neighbourhood, property type, living area, rooms, build and renovation year, garage, distance to centre, crime rate, school rating, sale price, and others). Because the data is synthetic, the findings illustrate the methods and should not be read as statements about a real market.

## Notebooks

| # | Topic | Methods | Highlights |
|---|-------|---------|------------|
| 01 | Regression | Random Forest, Gradient Boosting, Stacking | Predicting sale price. Best models reach R² = 0.89 (MAE ≈ 49,000) |
| 02 | Classification | Logistic Regression, Random Forest, Gradient Boosting, Stacking | Predicting a price class (Budget / Mid / Premium). Best F1 = 0.88 with Logistic Regression |
| 03 | Clustering | K-Means, Agglomerative, DBSCAN | Four property segments; K-Means had the best internal metrics |
| 04 | Dimensionality reduction | PCA, t-SNE, UMAP, TruncatedSVD | 60 encoded features: PCA keeps 95% of variance with 31 components |
| 05 | Pattern mining | Apriori, FP-Growth, association rules | Rules such as *large area + central location → high price* (lift 2.6) |
| 06 | Anomaly detection | Z-score, IQR, Isolation Forest, LOF, One-Class SVM | Compares global and local anomalies and finds inconsistent records |

Metrics come from my own runs with a fixed random seed and may differ slightly with other library versions.

## Project structure

```
real-estate-ml-lab/
├── data/
│   └── real_estate_regression_dataset.csv
├── notebooks/
│   ├── 01_regression.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_dimensionality_reduction.ipynb
│   ├── 05_pattern_mining.ipynb
│   └── 06_anomaly_detection.ipynb
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/a-tusevscaia/real-estate-ml-lab.git
cd real-estate-ml-lab
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run the notebooks from inside the `notebooks/` folder so that the relative path to `../data/` works. Notebooks save intermediate files to `notebooks/` or `outputs/`, which are ignored by Git.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · UMAP · mlxtend · Jupyter

## Author

Anastasia — Master's student in Information Technologies in Data Analysis (ASEM).
