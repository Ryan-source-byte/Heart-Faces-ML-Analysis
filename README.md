# Clinical Risk & Face Clustering Lab

Two compact machine-learning case studies in one reproducible notebook: ensemble classification for heart-disease risk and unsupervised identity clustering for face images.

The project demonstrates how model selection, dimensionality reduction and evaluation change across tabular clinical data and high-dimensional image data.

## Results at a glance

### Heart-disease classification

An 80/20 stratified split is used on the processed UCI Cleveland dataset. Missing values are imputed inside the training pipeline and categorical variables are one-hot encoded without test-set leakage.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| AdaBoost | **0.902** | **0.867** | **0.929** | **0.897** |
| Random Forest | 0.869 | 0.812 | **0.929** | 0.867 |
| Soft Voting | 0.869 | 0.833 | 0.893 | 0.862 |
| Decision Tree | 0.770 | 0.719 | 0.821 | 0.767 |

AdaBoost produced the best F1 score in this split and only two false negatives. Because the test set contains 61 examples, the notebook treats small model differences as experimental rather than universal.

### Face clustering

PCA reduced each Olivetti face from 4,096 pixels to 123 components while retaining at least 95% of variance.

| Feature space | Silhouette | Adjusted Rand Index |
| --- | ---: | ---: |
| Original 4,096 pixels | 0.1512 | 0.4214 |
| PCA, 123 components | **0.1719** | **0.4488** |

PCA improved k-means modestly, indicating useful noise reduction without making identities cleanly separable.

## What the notebook covers

- leakage-safe preprocessing with scikit-learn pipelines
- decision-tree, Random Forest, AdaBoost and soft-voting comparison
- confusion matrices and grouped Random Forest feature importance
- eigenface visualisation and explained-variance analysis
- k-means comparison before and after PCA
- fixed random seeds for repeatable splits and clustering

## Reproduce

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook assignment2_solution.ipynb
~~~

The Cleveland data file is committed. scikit-learn downloads Olivetti Faces on first use, so that step requires network access.

## Responsible-use note

This is an educational modelling study, not a clinical decision system. The heart-disease results are not validated for diagnosis, treatment or deployment, and the face experiment is limited to benchmark clustering analysis.
