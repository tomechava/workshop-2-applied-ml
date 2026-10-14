# Workshop 2 — MLP desde cero en PyTorch: resistencia del concreto

**Curso:** ST1613 Applied Machine Learning — EAFIT  
**Integrantes:** Tomás Echavarría Gil · Daniel Arango  
**Competencia:** [Concrete Strength Regression (Kaggle)](https://www.kaggle.com/competitions/concrete-strength-regression)

Red neuronal fully connected implementada desde cero en PyTorch para predecir la resistencia a la compresión del concreto (`csMPa`, MPa) a partir de su composición y edad. Métrica oficial: **MSE**.

## Estructura

```
├── workshop2.md            # Enunciado
├── DEVELOPMENT-PLAN.md     # Plan de trabajo
├── requirements.txt
├── data/                   # CSV de Kaggle (no versionados)
├── reference/              # Notebooks de clase (MNIST, ASL)
├── notebooks/              # Notebook principal del taller
├── submissions/            # submission.csv
└── reports/                # Figuras y pantallazo de Kaggle
```

## Cómo reproducir

1. Descargar `train.csv`, `test.csv` y `sample_submission.csv` de la competencia en `data/`:
   ```bash
   kaggle competitions download -c concrete-strength-regression -p data
   ```
   (descomprimir si llega como `.zip`).
2. Instalar dependencias:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate        # PowerShell: .venv\Scripts\Activate.ps1
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
3. Abrir `notebooks/workshop2_concrete_mlp.ipynb` y ejecutar *Restart & Run All*.

## Resultados

| Modelo | MSE CV (5-fold) | MSE público Kaggle |
|---|---|---|
| _pendiente_ | | |
