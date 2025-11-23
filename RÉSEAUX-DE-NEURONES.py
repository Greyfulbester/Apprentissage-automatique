# --------------------------------
# 5️⃣ RÉSEAUX DE NEURONES (DEEP LEARNING)
# --------------------------------
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
import numpy as np

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential([
        Dense(10, activation='relu', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 🔧 Correction ici : conversion en numpy
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)

    model.fit(X, y, epochs=100, verbose=0)

    print("\n🔹 Réseau de neurones : prédiction pour [1,1] →", model.predict(np.array([[1, 1]], dtype=np.float32))[0][0])

except ImportError:
    print("\n⚠️ TensorFlow n'est pas installé : partie Deep Learning ignorée.")

print("\n✅ Exécution terminée avec succès !")
