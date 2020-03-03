import pandas as pd
from sklearn.manifold import MDS
from pandas import DataFrame
import numpy as np

X = pd.read_csv("Yousem_StrokeXs_UpToDate_Rad2010_100dim_50iter_10window_5min_TRUEsent.csv")

print(X.shape)

Names = X.iloc[:,0]
Vec = X.iloc[:,1:]
print(Names.shape)
print(Vec.shape)

embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(Vec)
X_transformed.shape

to_csv = pd.DataFrame(X_transformed).to_csv("transformed_2dembedding.csv", header=False)
print("done")