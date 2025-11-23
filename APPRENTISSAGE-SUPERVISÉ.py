# ----------------------------
# 1️⃣ APPRENTISSAGE SUPERVISÉ
# ----------------------------

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
import numpy as np

# --- Régression linéaire ---
X = [[50], [60], [70], [80]]
y = [150000, 180000, 210000, 240000]

model_lr = LinearRegression()
model_lr.fit(X, y)
print("\n🔹 Régression linéaire : prédiction pour 75m² →", model_lr.predict([[75]])[0])

# --- Arbre de décision ---
X = [[15], [20], [30], [45]]
y = ["mineur", "majeur", "majeur", "majeur"]

model_tree = DecisionTreeClassifier()
model_tree.fit(X, y)
print("🔹 Arbre de décision : prédiction pour âge 17 →", model_tree.predict([[17]])[0])

# --- k-Nearest Neighbors (k-NN) ---
X = [[1, 2], [2, 3], [3, 3], [6, 7], [7, 8]]
y = ['A', 'A', 'A', 'B', 'B']

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X, y)
print("🔹 k-NN : prédiction pour (5,5) →", model_knn.predict([[5, 5]])[0])

# --- Régression logistique ---
X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model_log = LogisticRegression()
model_log.fit(X, y)
print("🔹 Régression logistique : prédiction pour 2.5 →", model_log.predict([[2.5]])[0])