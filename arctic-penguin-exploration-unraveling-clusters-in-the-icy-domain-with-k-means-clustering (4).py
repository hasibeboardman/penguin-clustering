#!/usr/bin/env python
# coding: utf-8

# This project aims to distinguish different penguin species by analyzing data collected about penguins in Antarctica. Create groups based on various penguin characteristics (culmen length, culmen depth, flipper length, body weight and gender) using the K-means clustering method.

# In[44]:


# Import Reqired Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from gap_statistic import OptimalK

import warnings
warnings.filterwarnings('ignore')


# In[45]:


# loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()


# In[46]:


penguins_df.info()


# In[47]:


# summary statistics to help identify outliners
penguins_df.describe()


# In[48]:


# check for null values 
penguins_df.isnull().sum()


# In[49]:


penguins_df.boxplot()
plt.show()


# Median (Green Line in the Middle): The green line inside each box shows the median (median value of the data set).
# Box (IQR - Interquartile Range): The lower and upper edges of each box indicate the first quartile (Q1, lower edge) and third quartile (Q3, upper edge) values. The area inside the box represents the middle 50% (IQR) of the data.
# Bars (Whiskers): The lines extending from the top and bottom of the box, often called "whiskers", indicate the lower and upper bounds of the data (usually values within 1.5 IQR from Q1 and Q3). These extend to the largest and smallest values that are not outliers.

# In[50]:


# Drop missing values
penguins_df = penguins_df.dropna()

# Clean up outliers
penguins_clean = penguins_df.copy()
 # Processing Columns in a Loop:
for col in ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']:
    # Calculation of First and Third Quarter Values:
    Q1 = penguins_clean[col].quantile(0.25)
    Q3 = penguins_clean[col].quantile(0.75)
    # Calculating the Interquartile Range (IQR)
    IQR = Q3 - Q1
    # Creating the Outlier Filter
    filter_outliers = (penguins_clean[col] >= Q1 - 1.5 * IQR) & (penguins_clean[col] <= Q3 + 1.5 * IQR)
    # Filtering DataFrame with Non-Outlier Values:
    penguins_clean = penguins_clean.loc[filter_outliers]


penguins_clean


# DataFrame Copy: The line penguins_clean = penguins_df.copy() creates a copy of the original DataFrame. This allows us to operate on the original data without modifying it.
# 
# Processing Columns in Loop: The for col in ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']: loop is used to process the specified columns (penguins' beak length, beak depth, fin length and body weight) one by one.
# 
# Calculating First and Third Quartile Values: The lines Q1 = penguins_clean[col].quantile(0.25) and Q3 = penguins_clean[col].quantile(0.75) calculate the first quartile (Q1) and third quartile (Q3) values for each column. These values represent the 25% and 75% percentiles of the data.
# 
# Calculating the Interquartile Range (IQR): The line IQR = Q3 - Q1 calculates the interquartile range (IQR). IQR is the difference between Q3 and Q1 values and measures the dispersion of the data.
# 
# Creating the Outlier Filter: The line filter_outliers = (penguins_clean[col] >= Q1 - 1.5 * IQR) & (penguins_clean[col] <= Q3 + 1.5 * IQR) determines the non-outlier values for each column. Here, non-outliers are defined as values falling between Q1 - 1.5 * IQR and Q3 + 1.5 * IQR. This is a commonly used definition of an outlier.
# 
# Filter DataFrame with Non-Outliers: The line penguins_clean = penguins_clean.loc[filter_outliers] creates a new DataFrame with rows with non-outliers. This step is repeated for each column, resulting in outliers being purged from all specified columns.

# In[51]:


penguins_clean.boxplot()
plt.show()


# Many machine learning models, such as K-Means clustering, rely on mathematical calculations that require numerical input. Categorical data often comes as labels with no inherent ordinal relationships or numerical values (for example, "Male" and "Female" for gender). Each dummy variable represents the presence or absence (1 or 0) of a particular category. This clear separation ensures that the unique impact of each category can be assessed individually and accurately by the model.

# In[52]:


# Perform preprocessing steps on the dataset to create dummy variables
# Making the gender variable numeric
df = pd.get_dummies(penguins_clean).drop('sex_.',axis=1)


# In[53]:


# Identify categorical and numerical columns
categorical_cols = ['sex']
numerical_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Create a column transformer with OneHotEncoder for categorical data and StandardScaler for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])


# With this piece of code, both numerical and categorical columns of a data set are processed appropriately and can be given as input to a machine learning model. This preprocessing step is critical for training the model accurately and effectively on the data.

# In[54]:


# Fit and transform the preprocessor on the cleaned data
penguins_preprocessed = preprocessor.fit_transform(penguins_clean)


# In[55]:


# # Scaling
scaler = StandardScaler()
X = scaler.fit_transform(df)
penguins_preprocessed = pd.DataFrame(data=X,columns=df.columns)
penguins_preprocessed.head()


# In[56]:


# Perform PCA
pca = PCA()
penguins_pca = pca.fit_transform(penguins_preprocessed)

# Determine the number of components with an explained variance ratio above 10%
explained_variance_ratio = pca.explained_variance_ratio_
n_components = sum(explained_variance_ratio >= 0.10)

# Execute PCA with Determined Components
pca = PCA(n_components=n_components)
penguins_PCA = pca.fit_transform(penguins_preprocessed)


# The penguins_PCA DataFrame is a new dataset that has reduced the size of the original penguins_preprocessed dataset and contains the most important components. This reduced dataset can be used in machine learning models and often preserves the necessary information while reducing computational time.

# In[57]:


# Results

print("Number of components used:", n_components)
print("Explained variance ratios of components:", explained_variance_ratio)
print("Data after PCA:\n", penguins_PCA[:5])


# As a result of PCA analysis, the data set was reduced to a two-dimensional space and these two components explain approximately 86% of the data (51.97% + 34.42%). These two components can be used to train machine learning models or perform data visualization and can replace the original four-dimensional dataset. This is a way to preserve important information while reducing the size of the dataset.

# In[58]:


# Step 4: Apply K-Means Clustering and Determine the Optimal Number of Clusters

# Elbow method to determine the optimal number of clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(penguins_PCA)
    wcss.append(kmeans.inertia_)
    


# In[59]:


# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# The graph shows that WCSS generally decreases as the number of clusters increases, but after a point this decrease slows down significantly. This is called the “elbow point” and is often used to determine the optimal number of clusters because after this point additional clusters are unlikely to add meaningful distinction to the data set.

# In[60]:


# Assuming the optimal number of clusters is at the "elbow point" of the graph
# This point represents where the WCSS starts to decrease more slowly
# Let's determine it visually from the plot
n_cluster = 4  # Example value, needs to be determined from the plot

n_cluster


# In[61]:


print("Inertia values for different number of clusters:")
for i, wcss_value in enumerate(wcss):
    print(f"Number of clusters: {i+1}, Inertia (WCSS): {wcss_value}")

plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (WCSS)')
plt.title('Inertia (WCSS) Values for Different Number of Clusters')
plt.show()

plt.show()


# In[62]:


# Silhouette Score
silhouette_scores = []

for i in range(2, 11):  # silhouette_score needs at least 2 clusters to start
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(penguins_PCA)
    score = silhouette_score(penguins_PCA, kmeans.labels_)
    silhouette_scores.append(score)

# Print and plot Silhouette Scores
print("Silhouette Scores:")
for i, score in zip(range(2, 11), silhouette_scores):
    print(f"Number of clusters: {i}, Silhouette Score: {score}")

plt.figure()
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()


# In[63]:


# Davies-Bouldin Index
db_scores = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(penguins_PCA)
    score = davies_bouldin_score(penguins_PCA, kmeans.labels_)
    db_scores.append(score)

# Print and plot Davies-Bouldin Scores
print("Davies-Bouldin Scores:")
for i, score in zip(range(2, 11), db_scores):
    print(f"Number of clusters: {i}, Davies-Bouldin Score: {score}")

plt.figure()
plt.plot(range(2, 11), db_scores, marker='o')
plt.title("Davies-Bouldin Scores for Different Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Score")
plt.grid(True)
plt.show()


# In[66]:


# Gap Statistics
optimalK = OptimalK(parallel_backend='none')
n_clusters = optimalK(penguins_PCA, cluster_array=np.arange(1, 11))
print(f"Optimal number of clusters according to Gap Statistics: {n_clusters}")


# In[67]:


# Integrated Analysis


# In[68]:


n_clusters=4


# In[69]:


# Run the k-means clustering algorithm with the optimal number of clusters.
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(penguins_PCA)


# In[70]:


# Visualize the resulting clusters
plt.scatter(penguins_PCA[:, 0], penguins_PCA[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title(f'K-means Clustering (K={n_clusters})')
plt.show()


# In[71]:


# penguins_clean DataFrame'inin sütun adlarını listeleme
print(penguins_clean.columns)


# In[72]:


# Create a final statistical DataFrame for each cluster.
# Adding cluster labels to the cleaned data
penguins_clean['label'] = kmeans.labels_
# Calculating the mean for each numeric column grouped by the cluster label
numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm','label']
stat_penguins = penguins_clean[numeric_columns].groupby('label').mean()
# Displaying the statistical summary table
stat_penguins


# In[73]:


# Assuming 'penguins_pca' is your PCA-transformed data and 'kmeans.labels_' are your cluster labels
plt.figure(figsize=(10, 8))
plt.scatter(penguins_pca[:, 0], penguins_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Penguin Data: PCA and K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




