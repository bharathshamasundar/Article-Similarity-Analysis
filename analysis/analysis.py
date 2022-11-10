#Importing libraries that are needed for processing

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from sklearn import linear_model

# Reading the csv file of term document frequency generated in R
input_data = pd.read_csv("finamatrix.csv",index_col=0, header =0 )


# Normalizing the data within a scale range, the input data is of the size 18828 x 100
x = preprocessing.MinMaxScaler()
data_standard = x.fit_transform(input_data)

# Calculating Cosine Similarity, the numpy array is of shape 18828 x 18828
cos_similarity = cosine_similarity(data_standard)

cos_similarity = cos_similarity[:100,:100] #sub-sampling to reduce dimension space

# Plotting the cosine heatmap
plt.title("Cosine Heatmap")

plt.imshow(cos_similarity, cmap= 'hot', interpolation= 'nearest')

plt.show()

euc_similarity = euclidean_distances(data_standard) #Calculating Euclidean distances first

euc_similarity = (1/(1+euc_similarity)) #Calculating  Euclidean similarity

euc_similarity = euc_similarity[:100,:100] # sub-sampling to reduce dimension space


# Plotting the euclidean heatmap
plt.title("Euclidean heatmap")
plt.imshow(euc_similarity, cmap='hot', interpolation= 'nearest')
plt.show()

# Calculating Jaccardian Distance
jaccard_similarity = pairwise_distances(data_standard, metric='jaccard')

jaccard_similarity = 1 - jaccard_similarity  # Jaccard similarity = 1 - Jaccard distance

jaccard_similarity = jaccard_similarity[:100,:100]  # reducing dimension

# Plotting Jaccardian heatmap
plt.title("Jaccard Heatmap")
plt.imshow(jaccard_similarity, cmap='hot', interpolation='nearest')
plt.show()

# Reshaping the array to 1000 x 1 from 100 x 100 to fit the linear curve
cos_similarity = cos_similarity.reshape(cos_similarity.size,1)
euc_similarity = euc_similarity.reshape(euc_similarity.size,1)
jaccard_similarity = jaccard_similarity.reshape(jaccard_similarity.size,1)

# Splitting test/train data for cosine matrix
cos_train = cos_similarity[:-500]
cos_test = cos_similarity[-500:]

# Splitting test/train data for euclidean matrix
euc_train = euc_similarity[:-500]
euc_test = euc_similarity[-500:]

# splitting test/train data for jaccardian matrix
jacc_train = jaccard_similarity[:-500]
jacc_test = jaccard_similarity[-500:]

# instantiating linear regression model
x = linear_model.LinearRegression()

# fitting the model y = ax + b where y=cosine and x= euclidean
x.fit(euc_train,cos_train)

# plotting the cosine euclidean fit scatter plot
plt.scatter(euc_test,cos_test, color = 'black')
plt.title('Cosine -Euclidean Fit')
plt.xlabel('Euclidean')
plt.ylabel('Cosine')
plt.xticks(())
plt.yticks(())
plt.plot(euc_test, x.predict(euc_test), color='blue', linewidth=3)

plt.show()

# fitting the model y = ax + b where y= euclidean and x= jaccardian
x.fit(jacc_train,euc_train)

#plotting the euclidian jacardian fit
plt.scatter(jacc_test,euc_test, color = 'black')
plt.title('Euclidian -Jacardian Fit')
plt.xlabel('Jacarrd')
plt.ylabel('Euclidean')
plt.xticks(())
plt.yticks(())
plt.plot(jacc_test, x.predict(jacc_test), color='green', linewidth=3)
plt.show()

# fitting the model y = ax + b where y= jaccardian and x= cosine
x.fit(cos_train,jacc_train)

#plotting the jaccard cosine fit
plt.scatter(cos_test,jacc_test, color = 'red')
plt.title('Jaccard-Cosine Fit')
plt.xlabel('Cosine')
plt.ylabel('Jaccard')
plt.xticks(())
plt.yticks(())
plt.plot(cos_test, x.predict(cos_test), color='black', linewidth=3)
plt.show()