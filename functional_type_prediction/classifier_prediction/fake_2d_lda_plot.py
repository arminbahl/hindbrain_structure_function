import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
np.random.seed(seed=42)
# Generate synthetic dataset with 10 features and 4 classes
X, y = make_classification(n_samples=131, n_features=13, n_informative=5, n_redundant=8,
                           n_clusters_per_class=1, n_classes=4, class_sep=5, random_state=1)
noise = np.random.normal(np.random.random(),np.random.randint(1,10), X.shape)
noise0 = np.random.normal(np.random.randint(1,6),np.random.randint(1,5), X[y==0].shape)
noise1 = np.random.normal(np.random.randint(1,6),np.random.randint(1,5), X[y==1].shape)
noise2 = np.random.normal(np.random.randint(1,6),np.random.randint(1,5), X[y==2].shape)
noise3 = np.random.normal(np.random.randint(1,6),np.random.randint(1,5), X[y==3].shape)
X = X+noise
X[y==0] = X[y==0]+noise0
X[y==1] = X[y==1]+noise1
X[y==2] = X[y==2]+noise2
X[y==3] = X[y==3]+noise3
# Fit LDA and project to 2D
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit_transform(X, y)  # Project data to 2D

# Train a simple classifier (e.g., LogisticRegression) on the 2D projected data
classifier = LogisticRegression()
classifier.fit(X_r2, y)

# Create a mesh grid for plotting decision boundaries in the 2D projected space
x_min, x_max = X_r2[:, 0].min() - 1, X_r2[:, 0].max() + 1
y_min, y_max = X_r2[:, 1].min() - 1, X_r2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                     np.arange(y_min, y_max, 0.001))

# Predict the class on the grid using the 2D classifier
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,10))
# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('#feb326b3', '#e84d8ab3', '#64c5ebb3', '#7f58afb3')))

# Plot the projected data points
scatter = plt.scatter(X_r2[:, 0], X_r2[:, 1], c=y, edgecolors='k', cmap=ListedColormap(('#feb326b3', '#e84d8ab3', '#64c5ebb3', '#7f58afb3')),s=300)
plt.xlabel('LD1')
plt.ylabel('LD2')
from pathlib import Path
import os
path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')
savepath = path_to_data / 'make_figures_FK_output' / '2D_LDA_PROJECTION'
os.makedirs(savepath, exist_ok=True)
plt.savefig(savepath / 'made up data lda example',dpi=300)
plt.savefig(savepath / 'made up data lda example.pdf',dpi=300)

plt.show()
