
# --------------------------------
# 3️⃣ APPRENTISSAGE SEMI-SUPERVISÉ
# --------------------------------
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
import numpy as np

X = [[0], [1], [2], [3], [4]]
y = [0, 1, -1, -1, -1]  # -1 = non étiqueté

model_label = LabelPropagation()
model_label.fit(X, y)
print("\n🔹 Apprentissage semi-supervisé : étiquettes finales →", model_label.transduction_)
