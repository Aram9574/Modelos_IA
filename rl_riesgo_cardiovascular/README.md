# ❤️ Predicción de riesgo cardiovascular a 10 años (Framingham)

> Proyecto de Machine Learning para estimar el riesgo de enfermedad cardiovascular a 10 años usando el **Framingham Heart Study Dataset**.

## 🎯 Objetivo
Construir y evaluar un modelo de clasificación que **priorice la detección temprana (recall)** de pacientes en riesgo, asumiendo un mayor número de falsos positivos por tratarse de un caso de **medicina preventiva**.

## 🔁 Pipeline seguido
1. **Exploración y análisis** con Pandas (balance de clases, `info()`, `describe()`, correlaciones).
2. **Preprocesamiento**  
   - Imputación de faltantes con **mediana** (robusta a outliers).  
   - Estandarización con **StandardScaler**.  
3. **Selección de características**  
   - **PCA** (Análisis de Componentes Principales) hasta **95%** de varianza explicada → submuestra depurada.  
4. **Entrenamiento y validación**  
   - `train_test_split` estratificado (80/20).  
   - Comparativa de escenarios: desbalanceado (umbral 0.5), balanceado (umbral 0.5) y balanceado (umbral **0.3**).  
   - Comparativa de algoritmos (LR, RF, GB, SVM, KNN).  
5. **Modelo seleccionado**  
   - **Regresión Logística** con `class_weight="balanced"` y **umbral 0.3**.

## 📊 Resultados clave (validación)
- **Recall (clase positiva)** ≈ **0.92**  
- **AUC** ≈ **0.70**  
- **Precision** ≈ 0.19  
- **Accuracy** ≈ 0.42  
> Trade-off intencional: maximizar sensibilidad para **no dejar pasar** pacientes en riesgo.

## 🗂️ Estructura del proyecto
riesgo_cardiovascular/
├── README.md
├── riesgo_cardiovascular.ipynb # Notebook principal (Google Colab/Jupyter)
├── rl_riesgo_cardiovascular.py # Script opcional con el pipeline
├── figures/
│ └── resumen_validacion.png # Matriz de confusión + ROC del modelo final
└── data/
└── framingham.csv # (opcional) dataset local si se distribuye


## ⚙️ Requisitos
- Python 3.10+
- `pip install -r ../requirements.txt` (o instalar: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter)

## ▶️ Cómo ejecutar
**Opción A – Notebook**
1. Abrir `riesgo_cardiovascular.ipynb` en Jupyter/Colab.  
2. Ejecutar celdas en orden (genera `figures/resumen_validacion.png`).

**Opción B – Script**
```bash
python rl_riesgo_cardiovascular.py
