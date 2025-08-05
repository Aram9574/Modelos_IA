# Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Cargar el dataset 
df = pd.read_csv("Breast_cancer_dataset.csv")

# Ver las primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Información general del dataset
print("\nResumen de columnas, tipos de datos y valores nulos:")
print(df.info())

# Estadísticas básicas
print("\nEstadísticas descriptivas:")
print(df.describe())


# Visualizar distribución de la variable objetivo
# Estilo bonito para los gráficos
sns.set(style="whitegrid")

# Gráfico de conteo de diagnósticos
sns.countplot(x="diagnosis", data=df)
plt.title("Distribución de diagnósticos (B = benigno, M = maligno)")
plt.xlabel("Diagnóstico")
plt.ylabel("Cantidad de casos")
plt.show()

# Convertir 'diagnosis' a valores numéricos: M = 1, B = 0
#df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})


# Mapa de correlación 
# Calcular la matriz de correlación
corr_matrix = df.corr(numeric_only=True)

# Mostrar el mapa de calor
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Matriz de correlación entre variables")
plt.show()

# Preprocesamiento (eliiminar, seleccionar, normalizar, dividir)
# X: variables predictoras (todas menos diagnosis)
X = df.drop(columns=["diagnosis"])

# y: variable objetivo
y = df["diagnosis"]

# Normalizar las variables predictoras
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#Verificar tamaños de los conjuntos
print("\nTamaños de los conjuntos de datos:")
print("Tamaño total del dataset:", df.shape)
print("Tamaño del set de entrenamiento:", X_train.shape)
print("Tamaño del set de prueba:", X_test.shape)


#Entrenamiento del modelo
# Crear el modelo
model = LogisticRegression(max_iter=1000)

# Entrenar (fit)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
print("\nResultados del modelo:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precisión:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()
