# Import the modules and Read the data.
import pandas as pd             
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/penguin.csv')

# Remove the missing values from the DataFrame using 'dropna()' function.
df.dropna(inplace=True)
# Add a new column 'label' that resembles 'species' column.
df['label'] = df['species']
df['label'] = df['label'].map({'Adelie':0, 'Chinstrap':1, 'Gentoo':2})
# Convert the non-numeric column 'sex' to numeric in the DataFrame.
df['sex'] = df['sex'].map({'Male':0, 'Female':1})
# Convert the non-numeric column 'island' to numeric in the DataFrame.
df['island'] = df['island'].map({'Biscoe':0, 'Dream':1, 'Torgersen':2})

# Create a DataFrame having only feature variables
features = df.drop(columns=['species','label'])

# Normalise the column values.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
features_scaled_df

# Transform the scaled data into 2D using PCA
# Pass the number of components to the 'PCA' constructor
pca = PCA(n_components=2)
# Call the 'fit_transform()' function on the PCA object.
pca_trans = pca.fit_transform(features_scaled_df)
# Print the 2D array.
pca_trans

# Convert 2D array to pandas DataFrame
pca_df = pd.DataFrame(pca_trans, columns=['pca1','pca2'])

import matplotlib.pyplot as plt
plt.figure(figsize=(20,6))
plt.scatter(pca_df['pca1'], pca_df['pca2'], c=df['label'])
plt.show()
