#Importing libraries that are needed for processing
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

# Reading the csv file of term document frequency generated in R
input_data = pd.read_csv("finamatrix.csv",index_col=0, header =0 )

# Normalizing the data within a scale range
x = preprocessing.MinMaxScaler()
data_standard = x.fit_transform(input_data)

# Selection data for 10, 30, 60 and 80 features
data_10 = data_standard[:,:10]

data_30 = data_standard[:,:30]

data_60 = data_standard[:,:60]

data_80 = data_standard[:,:80]

# Calculating cosine similarity for 10, 30, 60 80 and 100 features
print("The standard deviation of cosine similarities for 10, 30, 60, 80 and 100 features are as follows")

print(np.std(cosine_similarity(data_10)))

print(np.std(cosine_similarity(data_30)))

print(np.std(cosine_similarity(data_60)))

print(np.std(cosine_similarity(data_80)))

print(np.std(cosine_similarity(data_standard)))

# Calculating euclidean similarity for 10, 30, 60, 80 and 100 features

print("The standard deviation of Euclidean similarities for 10, 30, 60, 80 and 100 features are as follows")
euc_similarity = euclidean_distances(data_10)
euc_similarity = (1/(1+euc_similarity))
print(np.std(euc_similarity))

euc_similarity = euclidean_distances(data_30)
euc_similarity = (1/(1+euc_similarity))
print(np.std(euc_similarity))

euc_similarity = euclidean_distances(data_60)
euc_similarity = (1/(1+euc_similarity))
print(np.std(euc_similarity))

euc_similarity = euclidean_distances(data_80)
euc_similarity = (1/(1+euc_similarity))
print(np.std(euc_similarity))

euc_similarity = euclidean_distances(data_standard)
euc_similarity = (1/(1+euc_similarity))
print(np.std(euc_similarity))


# Calculating jaccardian similarity for 10, 30, 60, 80 and 100 features

print("the standard deviation of jaccardian similarites for 10, 30, 60, 80 and 100 features are as follows")
jac_similarity = pairwise_distances(data_10, metric='jaccard')
jac_similarity = 1- jac_similarity
print(np.nanstd(jac_similarity)) # using nanstd as there are some nAn numbers here


jac_similarity = pairwise_distances(data_30, metric='jaccard')
jac_similarity = 1- jac_similarity
print(np.nanstd(jac_similarity))

jac_similarity = pairwise_distances(data_60, metric='jaccard')
jac_similarity = 1- jac_similarity
print(np.nanstd(jac_similarity))

jac_similarity = pairwise_distances(data_80, metric='jaccard')
jac_similarity = 1- jac_similarity
print(np.nanstd(jac_similarity))

jac_similarity = pairwise_distances(data_standard, metric='jaccard')
jac_similarity = 1- jac_similarity
print(np.nanstd(jac_similarity))