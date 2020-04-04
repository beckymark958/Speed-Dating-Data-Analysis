import pandas as pd
#from sklearn.cluster import KMeans

df = pd.read_csv("google_review_ratings.csv", nrows = 5456)

#df.head()
#print(df.head())
#df.tail()
#print(df.tail())
test_dict = {'theatres': list(df['Category 6']), 'museums': list(df['Category 7'])}

print(test_dict)

#df.info()

#kmeans = KMeans(n_clusters = 6)
#kmeans.fit()


