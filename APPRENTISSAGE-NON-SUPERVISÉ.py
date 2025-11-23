# --------------------------------
# 2️⃣ APPRENTISSAGE NON SUPERVISÉ
# --------------------------------

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
import numpy as np

# --- K-Means (clustering) ---
X = np.array([[1, 2], [2, 1], [8, 9], [9, 8]])
model_kmeans = KMeans(n_clusters=2, random_state=0)
model_kmeans.fit(X)
print("\n🔹 K-Means : groupes attribués →", model_kmeans.labels_)

# --- PCA (réduction de dimension) ---
X = np.random.rand(5, 4)
pca = PCA(n_components=2)
X_reduit = pca.fit_transform(X)
print("🔹 PCA : données réduites →\n", X_reduit)