
# SVM Classification avec PCA et Optimisation des Hyperparamètres

Ce projet utilise un modèle de **Support Vector Machine (SVM)** pour la classification des données de diabète, avec une exploration de la réduction de dimensions via **PCA (Principal Component Analysis)**, ainsi qu'une optimisation des hyperparamètres avec **GridSearchCV**.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les bibliothèques suivantes :

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `mlxtend` (pour la visualisation des frontières de décision)

Vous pouvez installer ces bibliothèques via pip :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
```

## Description du Code

### 1. Prétraitement des Données

#### a) Chargement des Données
Le jeu de données est téléchargé via `kagglehub.dataset_download` et chargé dans un DataFrame pandas. Il contient des informations sur des patients, avec une colonne cible `Outcome` qui indique si le patient a le diabète (1) ou non (0).

```python
data = pd.read_csv(path + "/diabetes.csv")
X = data.drop(columns=["Outcome"]).values
Y = data["Outcome"].values
```

#### b) Réduction de Dimensions avec PCA
Une réduction de dimensionnalité est réalisée sur les données avec **PCA** afin de permettre une visualisation en 2D des données.

```python
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
```

Une visualisation de la projection des données sur les 2 premières composantes principales est générée.

```python
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=Y, palette="coolwarm", alpha=0.7)
```

### 2. Séparation du Jeu de Données

Le jeu de données est divisé en un ensemble d'entraînement (70 %) et un ensemble de test (30 %) pour entraîner et évaluer le modèle.

```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
```

Les données sont normalisées à l'aide de `StandardScaler` pour garantir que toutes les caractéristiques aient la même échelle.

### 3. Modélisation avec SVM

Deux modèles SVM sont entraînés : un modèle avec un noyau **RBF** (Radial Basis Function) et un modèle avec un noyau **linéaire**.

#### a) SVM RBF
Un SVM avec un noyau RBF est utilisé pour classer les données, et l'accuracy du modèle est affichée.

```python
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred = svm_rbf.predict(X_test)
```

#### b) SVM Linéaire
Un autre modèle SVM avec un noyau linéaire est également entraîné pour comparer les résultats.

```python
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
```

### 4. Optimisation des Hyperparamètres

L'optimisation des hyperparamètres est réalisée à l'aide de **GridSearchCV** pour trouver les meilleurs paramètres pour le modèle SVM.

```python
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

Les résultats du **GridSearchCV** sont comparés avec le modèle de base sans optimisation des hyperparamètres.

### 5. Analyse des Vecteurs Supports

Les vecteurs supports sont les points de données qui définissent la frontière de décision. Ils sont affichés pour chaque modèle.

```python
print(f"Nombre de vecteurs supports par classe(RBF): {svm_rbf.n_support_}")
print(f"Nombre de vecteurs supports par classe(Linéaire): {svm_linear.n_support_}")
```

### 6. Visualisation des Frontières de Décision

Les frontières de décision des modèles SVM sont visualisées après réduction des dimensions avec PCA. La fonction `plot_decision_boundary_8D_to_2D` génère cette visualisation.

```python
plot_decision_boundary_8D_to_2D(svm_rbf, X_train, X_pca_train, y_train, "SVM RBF (Projection 8D → 2D)", "SVM_RBF.png")
plot_decision_boundary_8D_to_2D(svm_linear, X_train, X_pca_train, y_train, "SVM Linéaire (Projection 8D → 2D)", "SVM_Linéaire.png")
```

### 7. Évaluation du Modèle

Les performances des modèles SVM sont évaluées avec le rapport de classification et l'accuracy. Ce processus est effectué pour les modèles SVM RBF et linéaire.

```python
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print(classification_report(y_test, y_pred_linear))
print(f"Accuracy: {accuracy_score(y_test, y_pred_linear)}")
```

## Résultats

Les performances des modèles SVM (RBF et linéaire) sont comparées à l'aide de différentes métriques d'évaluation, telles que l'accuracy, le rapport de classification, et la visualisation des frontières de décision.

## Conclusion

Ce projet permet de comprendre l'utilisation de SVM pour la classification et explore les techniques de réduction de dimensionnalité avec PCA ainsi que l'optimisation des hyperparamètres avec **GridSearchCV**.

## Références

- [Support Vector Machine - Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
- [PCA - Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
