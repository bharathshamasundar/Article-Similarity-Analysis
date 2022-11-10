#Importing libraries that are needed for processing
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Reading the csv file of term document frequency generated in R
input_data = pd.read_csv("finamatrix.csv",index_col=0, header =0 )

# Normalizing the data within a scale range
x = preprocessing.MinMaxScaler()
data_standard = x.fit_transform(input_data)

# Calculating Cosine Similarity
cos_similarity = cosine_similarity(data_standard)

# Reshaping numpy array to from N x N to N*N x 1
cos_similarity = cos_similarity.reshape(cos_similarity.size,1)

# Decrease sort
sorted_cos = np.sort(cos_similarity, axis = None)[::-1]

# Selecting score for the top three articles
top_cosine = sorted_cos[0:3]
print("top three pairs of articles have a cosine similarity score of")

print(top_cosine)

