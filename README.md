# Penguin Clustering with K-Means

This project applies unsupervised machine learning techniques to cluster different species of penguins using their physical characteristics. The dataset includes features such as culmen length, culmen depth, flipper length, body mass, and gender.

## Technologies Used
- Python 3
- pandas
- matplotlib
- scikit-learn (KMeans, PCA, StandardScaler)
- OneHotEncoder & ColumnTransformer
- OptimalK (gap statistic)
- Jupyter Notebook

## Dataset
The dataset used is `"penguins.csv"`, which contains data about penguin species found in Antarctica. Please ensure the CSV file is placed in the same directory.

## Project Stages
1. **Data Cleaning**: Null values and outliers are removed using IQR method.
2. **Feature Engineering**: Gender column is converted to dummy variables; features are scaled.
3. **Dimensionality Reduction**: PCA is used to reduce feature dimensions for clustering.
4. **Clustering**:
   - Elbow Method
   - Silhouette Score
   - Davies-Bouldin Index
   - Gap Statistic
5. **Visualization**: Final clusters are visualized using PCA components.

## How to Run
Make sure you have the required packages installed:
```bash
pip install pandas matplotlib scikit-learn gap-stat
```

