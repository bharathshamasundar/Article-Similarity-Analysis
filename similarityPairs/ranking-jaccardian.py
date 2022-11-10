#Importing libraries that are needed for processing
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances



# Reading the csv file of term document frequency generated in R
input_data = pd.read_csv("finamatrix.csv",index_col=0, header =0 )

# Normalizing the data within a scale range
x = preprocessing.MinMaxScaler()
data_standard = x.fit_transform(input_data)

# Calculating jaccardian similarity scores

jac_similarity = pairwise_distances(data_standard, metric='jaccard')

# Reshaping array
jac_similarity = jac_similarity.reshape(jac_similarity.size,1)

print jac_similarity[0:10]
# sorting scores
sorted_jac = np.sort(jac_similarity, axis = None)[::-1]

sorted_jac = np.roll(sorted_jac, -np.count_nonzero(np.isnan(jac_similarity)))
#Top three paired article scores are
top_jac = sorted_jac[0:3]

print sorted_jac[0:20]

print("top three jacardian score paired articles are")

print(top_jac)
