# 🤖 Modelos de Machine Learning en Salud

Este repositorio recopila diferentes proyectos de **Machine Learning aplicados a la salud**, desarrollados en **Python** y organizados por carpetas según el caso de estudio.  
Cada proyecto incluye un *notebook* de Google Colab/Jupyter, código fuente en `.py` y, en algunos casos, los datasets utilizados.

## 📂 Estructura del repositorio

- `cancer_mama/` → Modelo de clasificación para predecir cáncer de mama (Regresión Logística).  
- `riesgo_cardiovascular/` → Modelo de predicción de riesgo cardiovascular a 10 años (Regresión Logística con PCA y ajuste de umbral).  
- *(Próximos proyectos se agregarán en nuevas carpetas con la misma estructura).*

## 🚀 Tecnologías utilizadas

- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn (modelos, métricas, preprocesamiento, PCA)  

## 🧩 Flujo general de los proyectos

Cada proyecto sigue un pipeline común:  

1. **Carga y exploración del dataset**  
   - Limpieza y análisis descriptivo con Pandas.  
   - Visualización de distribuciones y correlaciones.  

2. **Preprocesamiento**  
   - Imputación de datos faltantes.  
   - Normalización y/o estandarización de variables.  
   - Reducción de dimensionalidad (ej. PCA).  

3. **Entrenamiento y validación**  
   - División estratificada en train/validación.  
   - Entrenamiento de uno o varios modelos de Scikit-Learn.  
   - Comparativa de algoritmos y escenarios (umbral, balanceo, etc.).  

4. **Evaluación**  
   - Métricas de desempeño (Accuracy, Precision, Recall, F1-score, AUC).  
   - Matrices de confusión.  
   - Curvas ROC.  

## 📊 Resultados actuales

- **Cáncer de mama** → Regresión Logística con accuracy ≈ 97%.  
- **Riesgo cardiovascular** → Regresión Logística balanceada con umbral 0.3, recall ≈ 92%, priorizando detección temprana en medicina preventiva.  

## 📌 Objetivo del repositorio

Construir una colección de proyectos de aprendizaje automático aplicados al ámbito de la salud que:  
- Sirvan como **portafolio académico y profesional**.  
- Permitan comparar diferentes algoritmos en distintos contextos.  
- Sean fácilmente ampliables con nuevos casos de uso.  

## 🛠️ Cómo usar

1. Clona este repositorio:  
   ```bash
   git clone https://github.com/Aram9574/Modelos_IA.git
   cd Modelos_IA
