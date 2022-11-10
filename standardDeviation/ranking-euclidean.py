import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


#Reading the csv file of term document frequency generated in R
input_data = pd.read_csv("finamatrix.csv",index_col=0, header =0 )


#Normalizing the data within a scale range
x = preprocessing.MinMaxScaler()
data_standard = x.fit_transform(input_data)

# Calculating Euclidean Similarity
euc_similarity = euclidean_distances(data_standard)
euc_similarity = (1/(1+euc_similarity))

# Reshaping numpy array to from N x N to N*N x 1
euc_similarity = euc_similarity.reshape(euc_similarity.size,1)

# Decrease sort
sorted_euc = np.sort(euc_similarity, axis = None)[::-1]

# Selecting score for the top three articles
top_euc = sorted_euc[0:3]

print("top three pairs of articles have a euclidean similarity score of")

print(top_euc)

