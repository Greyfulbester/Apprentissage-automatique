# --------------------------------
# 4️⃣ APPRENTISSAGE PAR RENFORCEMENT
# --------------------------------
# Exemple simplifié de Q-Learning (illustration seulement)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
import numpy as np

R = np.array([
    [-1, 0, -1],
    [0, -1, 1],
    [-1, 1, -1]
])
Q = np.zeros_like(R, dtype=float)
gamma = 0.8

for _ in range(1000):
    state = np.random.randint(0, 3)
    actions = np.where(R[state] >= 0)[0]
    if len(actions) == 0:
        continue
    action = np.random.choice(actions)
    next_state = action
    Q[state, action] = R[state, action] + gamma * np.max(Q[next_state])

print("\n🔹 Q-Learning : table Q finale →\n", Q)
