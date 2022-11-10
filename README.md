#Bharath Shamasundar


# Article-Similarity-Search


This is a four stage project of finding the similarity scores between article pairs from the 20newsgroup dataset

#Data Pre-Processing in R

All the files related to data processing are present in the dataProcessing folder.

The R notebook contains the data pre-processing part. Raw data of the 20newsgroup data is read, processed and stored as a term frequency matrix. In the tf matrix the rows are the document/article numbers and the colums are the terms that are present in each of these documents. Each entry in the tf matrix is the frequency of that term in the matrix.

Once the tf matrix has been generated the top 100 words or the best features are selected and this result in present in top100.csv(each word has a column frequency against itself). finmatrix.csv is the matrix of top 100 words against the terms frequency of all the documents.


#Similarity Analysis

Go to analysis folder and Run
```
python analysis.py
```

In this section the tf matrix now reduced to its top features, is operated upon to find the Cosine, Euclidean and Jaccardian similarity matrices respectively. It's heatmap analysis is also given.

The three similarity matrices are then analysed on the basis of Linear Regression to fit the similarity pairs using the equation:

Cosine=a*Euclidean+b

Euclidean=a*Jarcard+b

Jarcard=a*Cosine+b.

where a is the slope of the line and b is the y-intercept

#Standard Deviation Analysis

Go to the standardDeviation folder and Run
```
python Standard_deviation.py
```
The change to the SD of the similarity matrix with respect to the number of features is given.

#Sorted similarity pairs

Go to the similarityPairs folder and Run
```
python ranking-cosine.py
python ranking-euclidean.py
python ranking-jaccardian.py
```
The top 3 similarity pair scores for the articles of each metric are given here.
