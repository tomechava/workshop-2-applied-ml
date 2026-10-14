# DEVELOPMENT-PLAN — Workshop 2: MLP desde cero para resistencia del concreto

Plan de trabajo paso a paso, desde la creación del repositorio hasta la submission en Kaggle.
Basado en `workshop2.md`, los notebooks de referencia (`01_mnist`, `02_asl`) y una inspección previa de los CSV.

---

## 0. Hallazgos de la exploración inicial

### 0.1 Archivos encontrados

| Archivo | Contenido | Observación |
|---|---|---|
| `workshop2.md` | Enunciado | Rúbrica: EDA 20 %, Preproc. 20 %, PyTorch 25 %, Experimentos 20 %, Kaggle + conclusiones 15 % |
| `01_mnist.ipynb.ipynb` | `nn.Sequential`, `Adam`, `train()`, `validate()`, `get_batch_accuracy` | Doble extensión `.ipynb.ipynb` → renombrar a `01_mnist.ipynb` (el enunciado enlaza ese nombre) |
| `02_asl.ipynb.ipynb` | Datos tabulares con pandas, `MyDataset(Dataset)`, `DataLoader` | Doble extensión; tiene código de Google Colab (`drive.mount`) que no aplica localmente |
| `train.csv` | 721 × 10 | `Row ID` + 8 predictores + `csMPa` (target) |
| `test.csv` | 309 × 9 | `Row ID` 721–1029, sin target |
| `sample_submission.csv` | Columnas `Row ID,csMPa` | **Solo 10 filas ilustrativas** con IDs del rango de train → la submission real debe tener **309 filas** con los `Row ID` de `test.csv` (confirmar en la pestaña *Data* / *Overview* de Kaggle) |

### 0.2 Nombres reales de las columnas

```
Row ID, cement, slag, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age, csMPa
```

- `Row ID` → identificador, **no** es predictor (corr. con target ≈ −0.02).
- `csMPa` → target (MPa).

### 0.3 Datos clave (verificados)

| Hecho | Valor | Implicación |
|---|---|---|
| Valores faltantes | 0 en train y test | No se imputa; documentarlo |
| Tipos | Todo numérico (`age` es int) | No hay encoding; justificarlo |
| Duplicados exactos (sin `Row ID`) | 16 filas | Decidir: eliminar o conservar (y justificar) |
| Duplicados solo en features | 26 filas (7 mezclas con target distinto) | Ruido de laboratorio; usar **GroupKFold** o eliminar duplicados para que la misma mezcla no quede en train y valid a la vez (leakage) |
| Filas de test idénticas (features) a train | 11 | Curiosidad para el EDA; no explotar manualmente |
| Ceros | slag 322, flyash 387, superplasticizer 249 | Indicadores binarios `has_slag`, `has_flyash`, `has_sp` |
| `age` | skew 3.28; valores 28 (298), 3 (96), 7 (82), 56 (69)… | `log1p(age)` |
| Target `csMPa` | media 36.2, std 17.2, rango ~2–82.6, skew 0.39 | Ligeramente asimétrico; escalar con `StandardScaler` |
| Correlación con target | cement 0.52, superplasticizer 0.40, age 0.34, water −0.30 | Agua/cemento debería ser muy informativa |
| Train vs test | Medias y rangos muy similares | Sin *covariate shift* importante (mostrar con KDE/hist superpuestos) |

### 0.4 Entorno local

| Herramienta | Estado |
|---|---|
| Python + `torch 2.11.0+cpu` | ✅ instalado, **sin CUDA** (CPU basta para 721 filas) |
| `scikit-learn 1.8.0` | ✅ |
| `git 2.42` / `gh 2.101` | ✅ |
| `kaggle` CLI | ❌ no instalado → `pip install kaggle` (o subir por la web) |

> ⚠️ **No usar `torch.compile`** (aparece en los notebooks de clase): en Windows/CPU suele fallar o ser lento, y no aporta nada con un dataset tan pequeño.

---

## 1. Estructura objetivo del repositorio

```
workshop2/
├── README.md                    # Descripción, cómo correr, resultado en Kaggle
├── DEVELOPMENT-PLAN.md          # Este documento
├── workshop2.md                 # Enunciado
├── requirements.txt
├── .gitignore
├── data/                        # (ignorado por git) train.csv, test.csv, sample_submission.csv
├── reference/
│   ├── 01_mnist.ipynb
│   └── 02_asl.ipynb
├── notebooks/
│   └── workshop2_concrete_mlp.ipynb   # ENTREGABLE PRINCIPAL
├── submissions/
│   └── submission.csv           # ENTREGABLE
└── reports/
    ├── figures/                 # Curvas, EDA, etc. exportadas
    └── kaggle_score.png         # ENTREGABLE (pantallazo)
```

---

## 2. Fase 1 — Repositorio y entorno

### 2.1 Inicializar git y organizar archivos

```bash
cd "C:/Users/ASUS/OneDrive/Documents/EAFIT/SEMESTRE 9/Applied-ML/workshop2"
git init -b main
mkdir data reference notebooks submissions reports reports/figures
mv train.csv test.csv sample_submission.csv data/
mv 01_mnist.ipynb.ipynb reference/01_mnist.ipynb
mv 02_asl.ipynb.ipynb   reference/02_asl.ipynb
```

> Actualizar los enlaces de la tabla de `workshop2.md` a `reference/…` si se quiere que funcionen en GitHub.

### 2.2 `.gitignore`

```
# Datos de la competencia (las reglas de Kaggle suelen prohibir redistribuirlos)
data/
# Python / Jupyter
__pycache__/
.ipynb_checkpoints/
.venv/
*.pt
# Credenciales
kaggle.json
.env
```

> `submissions/submission.csv` **sí** se versiona (es un entregable). Si el profesor quiere ver los datos en el repo, quitar `data/` del ignore.

### 2.3 Entorno virtual y dependencias

```bash
python -m venv .venv
source .venv/Scripts/activate          # Git Bash  (PowerShell: .venv\Scripts\Activate.ps1)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel kaggle
pip freeze > requirements.txt
python -m ipykernel install --user --name workshop2 --display-name "Python (workshop2)"
```

(Alternativa: usar el Python global, que ya tiene torch y sklearn.)

### 2.4 Primer commit local y repo en GitHub

> **Regla del proyecto:** Claude solo hace `git add` / `git commit`. La creación del repo en GitHub, `git remote add` y **todo `git push` los ejecuta el usuario**.

```bash
git add .
git commit -m "chore: estructura inicial del workshop 2"
```

El usuario crea el repo **vacío** en GitHub (sin README, .gitignore ni licencia) y luego:

```bash
git remote add origin https://github.com/<usuario>/<repo>.git
git remote -v
git push -u origin main
```

- ✅ **Checkpoint:** repo visible en GitHub, sin los CSV, con README mínimo.

### 2.5 Credenciales de Kaggle

1. Kaggle → *Settings* → *API* → **Create New Token** → descarga `kaggle.json`.
2. Copiarlo a `C:\Users\ASUS\.kaggle\kaggle.json` (nunca al repo).
3. Aceptar las reglas en la página de la competencia (*Join Competition*).
4. Probar: `kaggle competitions files concrete-strength-regression`.

### Flujo de commits sugerido

Un commit (o rama + PR si se trabaja en equipo) por fase:
`feat: EDA` → `feat: preprocesamiento` → `feat: dataset y modelo` → `feat: entrenamiento` → `feat: experimentos K-Fold` → `feat: submission` → `docs: conclusiones y README`.

---

## 3. Fase 2 — Esqueleto del notebook

Crear `notebooks/workshop2_concrete_mlp.ipynb` con estas secciones (Markdown `##`), alineadas con la rúbrica:

0. Título, autores, objetivo, índice
1. Setup: imports, semillas, `device`, rutas
2. Parte 1 — EDA
3. Parte 2 — Preprocesamiento
4. Parte 3 — `Dataset` y `DataLoader`
5. Parte 4 — Arquitectura
6. Parte 5 — Entrenamiento y validación
7. Parte 6 — Experimentación (K-Fold + baselines)
8. Parte 7 — Reentrenamiento final y submission
9. Conclusiones

### Celda de setup

```python
import os, random, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data"
ID_COL, TARGET = "Row ID", "csMPa"
```

---

## 4. Fase 3 — Parte 1: EDA (20 %)

Checklist (cada ítem = celda de código + breve Markdown interpretando):

- [ ] 1. `shape`, `info()`, `head()` de train y test.
- [ ] 2. `describe().T`.
- [ ] 3. `isna().sum()` y duplicados: exactos (16) y por features (26, mostrar las 7 mezclas con target distinto).
- [ ] 4. Clasificar columnas: 8 numéricas, 0 categóricas, `Row ID` excluido (mostrar corr ≈ 0 como evidencia).
- [ ] 5. Target: histograma + KDE + boxplot; skew; outliers por IQR.
- [ ] 6. Histogramas de los 8 predictores (grilla 2×4); % de ceros por variable; `age.value_counts()` (bar plot).
- [ ] 7. Heatmap de correlación (Pearson **y** Spearman) + scatter de cada predictor vs `csMPa` (grilla 2×4). Agregar scatter de `water/cement` vs target para motivar el feature engineering.
- [ ] 8. Train vs test: histogramas/KDE superpuestos por variable (+ opcional test KS de `scipy.stats.ks_2samp`).
- [ ] Guardar figuras en `reports/figures/`.

### Hallazgos esperados (redactar 3–5 en Markdown)

1. **Escalas muy distintas** (superplasticizer 0–32 vs coarseaggregate 800–1145) → escalamiento obligatorio (StandardScaler).
2. **`age` muy sesgada** (skew 3.3) y concentrada en pocos valores → `log1p(age)`; relación no lineal con el target (crece rápido y se satura).
3. **Muchos ceros** en slag/flyash/superplasticizer (35–54 %) → indicadores binarios de presencia.
4. **Relación agua/cemento**: cement (+0.52) y water (−0.30) actúan en sentidos opuestos → feature `water/cement`.
5. **Sin faltantes ni categóricas, distribuciones train/test similares; hay duplicados** → no imputar ni codificar; manejar duplicados para evitar leakage en la validación.

---

## 5. Fase 4 — Parte 2: Preprocesamiento (20 %)

### 5.1 Decisiones

| Paso | Decisión | Justificación |
|---|---|---|
| Duplicados | Promediar el target de filas con features idénticas **o** usar `GroupKFold` con un id de grupo = hash de las features | Evita que la misma mezcla esté en train y valid |
| Faltantes | Ninguno → no imputar (mostrar `assert X.isna().sum().sum()==0`) | Confirmado |
| Categóricas | Ninguna → sin encoding. `age` como categórica: opcional, solo si se compara | Todas numéricas |
| Escalamiento X | `StandardScaler` (comparar con `MinMaxScaler` como experimento) | Red sensible a escala; las variables no tienen cotas duras |
| Escalamiento y | `StandardScaler` sobre `y` (reshape `(-1,1)`), **invertir** antes de métricas/submission | Estabiliza el entrenamiento |
| FE (set "fe") | `log_age = log1p(age)`, `w_c = water/cement`, `binder = cement+slag+flyash`, `w_b = water/binder`, `has_slag`, `has_flyash`, `has_sp` | Conocimiento del dominio + EDA |

### 5.2 Implementación sin leakage

```python
FEATURES_RAW = ["cement","slag","flyash","water","superplasticizer",
                "coarseaggregate","fineaggregate","age"]

def add_features(df):
    df = df.copy()
    df["log_age"]  = np.log1p(df["age"])
    df["w_c"]      = df["water"] / df["cement"]
    df["binder"]   = df["cement"] + df["slag"] + df["flyash"]
    df["w_b"]      = df["water"] / df["binder"]
    for c, n in [("slag","has_slag"),("flyash","has_flyash"),("superplasticizer","has_sp")]:
        df[n] = (df[c] > 0).astype(float)
    return df.drop(columns=["age"])
```

- `add_features` es **fila a fila** (sin estadísticas del dataset) → no filtra información; se puede aplicar antes de partir. Encapsularlo igualmente en `Pipeline([("fe", FunctionTransformer(add_features)), ("scaler", StandardScaler())])`.
- Partición 80/20 con `random_state=SEED` (para la parte "single split" y las curvas de aprendizaje).
- `pipe.fit(X_train)` → `transform(X_valid)`, `transform(X_test)`. Igual con `y_scaler`.
- Dentro de K-Fold: **crear y ajustar un pipeline nuevo por fold**.

---

## 6. Fase 5 — Partes 3, 4 y 5: PyTorch desde cero (25 %)

### 6.1 `Dataset` y `DataLoader` (adaptado de `MyDataset` de 02_asl)

```python
class ConcreteDataset(Dataset):
    def __init__(self, X, y=None):
        self.xs = torch.tensor(X, dtype=torch.float32).to(device)
        self.ys = None if y is None else torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    def __getitem__(self, idx):
        return (self.xs[idx],) if self.ys is None else (self.xs[idx], self.ys[idx])
    def __len__(self):
        return len(self.xs)

g = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
```

- Verificar `batch[0].shape == (B, n_features)` y `batch[1].shape == (B, 1)` (evita el broadcasting silencioso de `MSELoss`).

### 6.2 Arquitectura (≥ 3 capas ocultas, salida lineal de 1 neurona)

```python
def build_mlp(in_dim, hidden=(128, 64, 32), act=nn.ReLU, dropout=0.0, batchnorm=False):
    layers, prev = [], in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if batchnorm: layers.append(nn.BatchNorm1d(h))
        layers.append(act())
        if dropout: layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))          # sin activación
    return nn.Sequential(*layers).to(device)

model = build_mlp(X_train.shape[1])
print(model)
print("Parámetros entrenables:", sum(p.numel() for p in model.parameters() if p.requires_grad))
```

### 6.3 Entrenamiento / validación (adaptado de `train()`/`validate()`)

Cambios respecto a los notebooks de clase:
- `nn.MSELoss()` en lugar de `CrossEntropyLoss`.
- Pérdida **promediada por muestra** (`batch_loss.item() * len(x)` / N), no suma de batches.
- Funciones con parámetros (`model`, `loader`, `optimizer`) en vez de globales → reutilizables en K-Fold.
- `get_batch_accuracy` → función `regression_metrics(y_true, y_pred)` que devuelve MSE, RMSE, MAE, R² **en MPa** (tras `y_scaler.inverse_transform`).

```python
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train(); total = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward(); optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    return torch.cat([model(b[0]) for b in loader]).cpu().numpy()

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}

def fit(model, train_loader, valid_loader, y_valid_mpa, y_scaler, lr=1e-3,
        weight_decay=0.0, epochs=1000, patience=50):
    loss_fn = nn.MSELoss()
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best, best_state, wait = [], np.inf, None, 0
    for ep in range(epochs):
        tr_loss = train_one_epoch(model, train_loader, loss_fn, opt)
        pred_mpa = y_scaler.inverse_transform(predict(model, valid_loader))
        m = regression_metrics(y_valid_mpa, pred_mpa)
        history.append({"epoch": ep, "train_loss": tr_loss, **{f"val_{k}": v for k, v in m.items()}})
        if m["MSE"] < best - 1e-4:
            best, wait = m["MSE"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
```

- Registrar también el `val_loss` en escala escalada para graficar **train vs valid en la misma escala** (o graficar ambos en MPa²).
- Early stopping por MSE de validación con `patience` ≈ 50 y restauración del mejor estado.
- Opcional: `torch.optim.lr_scheduler.ReduceLROnPlateau`.

### 6.4 Entregables de esta fase

- [ ] Modelo base (p. ej. 3×[128,64,32], ReLU, Adam lr=1e-3, batch 32) entrenado en el split 80/20.
- [ ] Curva de aprendizaje (train vs valid loss por época, eje y log), marcar la época del mejor modelo.
- [ ] Tabla de MSE/RMSE/MAE/R² en validación (MPa).
- [ ] Scatter predicho vs real + gráfico de residuos.
- [ ] Discusión under/overfitting.
- [ ] Justificación del lr: 1e-3 es el default de Adam; opcional *LR sweep* rápido (1e-4, 3e-4, 1e-3, 3e-3, 1e-2).

**Referencia esperada:** en este dataset una MLP bien ajustada suele lograr RMSE ≈ 4–6 MPa (MSE ≈ 20–35, R² ≈ 0.88–0.94); una regresión lineal ronda RMSE ≈ 10 (MSE ≈ 100+).

---

## 7. Fase 6 — Parte 6: Experimentación (20 %)

### 7.1 Protocolo

- **K-Fold K=5** (idealmente `GroupKFold` por mezcla o `KFold(shuffle=True, random_state=SEED)` tras deduplicar), mismo esquema de folds para todos los experimentos.
- En cada fold: pipeline nuevo (fit solo en el fold de train) → entrenar con early stopping usando el fold de valid → registrar MSE (MPa²) + RMSE, MAE, R² + nº de épocas.
- Reportar **media ± std** por configuración.
- Función `run_cv(config) -> dict` que reciba un diccionario de hiperparámetros, para iterar fácilmente.

> Nota de rigor: usar el fold de valid tanto para early stopping como para reportar sesga levemente el resultado hacia abajo. Aceptable para el taller; mencionarlo. (Versión estricta: separar un 10–15 % interno del fold de train para early stopping.)

### 7.2 Baselines (mismos folds)

| Baseline | Cómo |
|---|---|
| Media | `DummyRegressor(strategy="mean")` |
| Lineal (raw) | `Pipeline([StandardScaler, LinearRegression])` |
| Lineal (+FE) | Con `add_features` |

### 7.3 Experimentos mínimos (≥ 3; se recomiendan 6–8)

| # | Variable que cambia | Configuraciones |
|---|---|---|
| E1 | Features | raw (8) vs raw + log_age vs FE completo |
| E2 | Profundidad/anchura | [64,32,16], [128,64,32], [256,128,64,32] |
| E3 | Activación | ReLU vs LeakyReLU vs GELU vs SiLU |
| E4 | Regularización | nada vs Dropout 0.1/0.2 vs BatchNorm vs weight_decay 1e-4/1e-3 |
| E5 | lr / batch size | lr ∈ {3e-4, 1e-3, 3e-3}; batch ∈ {16, 32, 64} |
| E6 | Escalamiento | StandardScaler vs MinMaxScaler; con vs sin escalar `y` |
| E7 (opc.) | Pérdida | MSELoss vs HuberLoss |
| E8 (opc.) | `age` categórica | one-hot de `age` vs `log1p(age)` (el enunciado pide comparar si se usa) |

Estrategia: búsqueda **secuencial/greedy** (fijar el mejor de E1, luego E2, etc.) para controlar el tiempo de cómputo en CPU.

### 7.4 Resultados

- [ ] Tabla comparativa `configuración → MSE val (media ± std), RMSE, MAE, R²`, ordenada por MSE, incluyendo baselines.
- [ ] Bar chart con barras de error.
- [ ] Guardar `reports/experiments.csv`.
- [ ] Análisis: ¿la red supera al baseline? ¿qué decisión tuvo más impacto?

---

## 8. Fase 7 — Parte 7: Modelo final y submission (15 %)

1. Elegir la mejor configuración por MSE medio de CV.
2. **Número de épocas para el reentrenamiento:** usar la mediana de las épocas óptimas de los folds (no hay validación al entrenar con todo `train.csv`).
3. Reentrenar sobre todo `train.csv` (pipeline ajustado en todo train).
   - Alternativa recomendada (más robusta): **ensamble** = promedio de las predicciones de los 5 modelos de los folds y/o de varias semillas.
4. Aplicar el **mismo pipeline** a `test.csv` → predecir → `y_scaler.inverse_transform`.
5. Construir y validar la submission:

```python
sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: preds.ravel()})
assert list(sub.columns) == list(pd.read_csv(f"{DATA_DIR}/sample_submission.csv").columns)
assert len(sub) == len(test) == 309
assert sub[TARGET].notna().all() and np.isfinite(sub[TARGET]).all()
sub.to_csv("../submissions/submission.csv", index=False)
```

6. Subir a Kaggle:

```bash
kaggle competitions submit -c concrete-strength-regression \
  -f submissions/submission.csv -m "MLP [128,64,32] GELU +FE, 5-fold ensemble"
kaggle competitions submissions -c concrete-strength-regression
```

(O por la web: *Submit Prediction* → subir CSV.)

7. Tomar pantallazo del **puntaje público** (leaderboard/submissions) → `reports/kaggle_score.png` e insertarlo en el notebook (`![score](../reports/kaggle_score.png)`).
8. Comparar el MSE público con el MSE de CV y comentar la diferencia.

> Revisar el límite diario de submissions; no gastar intentos sin una mejora clara en CV.

---

## 9. Fase 8 — Conclusiones, verificación y entrega

### 9.1 Conclusiones (1–2 párrafos al final del notebook)

- Qué funcionó (p. ej. FE con `w_c` y `log_age`, escalamiento de `y`, early stopping, ensamble).
- Qué no (p. ej. redes muy profundas sin regularización → overfitting; MinMax vs Standard sin diferencia).
- Comparación con baselines y puntaje en Kaggle.
- Con más tiempo: búsqueda de hiperparámetros con Optuna, más semillas, comparar con gradient boosting como referencia, modelar explícitamente la curva de madurez por edad.

### 9.2 Checklist de calidad antes de entregar

- [ ] *Kernel → Restart & Run All* sin errores (y en tiempo razonable en CPU; si tarda, reducir `epochs`/configs).
- [ ] Rutas relativas (`../data`), sin rutas de Colab/Drive.
- [ ] Semillas fijadas; resultados reproducibles.
- [ ] Ningún `fit` usa validación o test.
- [ ] Target en forma `(N,1)` y `float32`; métricas en MPa (target des-escalado).
- [ ] `model.train()` / `model.eval()` + `torch.no_grad()` correctos.
- [ ] Cada sección tiene Markdown explicando la decisión.
- [ ] Modelo impreso + nº de parámetros.
- [ ] Curvas de aprendizaje, tabla de experimentos, baselines.
- [ ] `submission.csv` con 309 filas y columnas `Row ID,csMPa`.
- [ ] Pantallazo de Kaggle incluido.
- [ ] README con: objetivo, estructura, cómo reproducir, mejor CV, puntaje público.

### 9.3 Cierre en GitHub

```bash
git add .
git commit -m "docs: conclusiones, submission y pantallazo de Kaggle"
git push                 # (lo ejecuta el usuario)
git tag v1.0-entrega && git push --tags   # (lo ejecuta el usuario)
```

Entregar según indique el curso: notebook (`.ipynb`), `submission.csv`, pantallazo y enlace al repo.

---

## 10. Orden de ejecución resumido

| Paso | Fase | Resultado |
|---|---|---|
| 1 | Repo + entorno + Kaggle API | Repo en GitHub, `kaggle` funcionando |
| 2 | Esqueleto del notebook | Secciones y setup |
| 3 | EDA | Figuras + 3–5 hallazgos |
| 4 | Preprocesamiento | Pipeline sin leakage |
| 5 | Dataset/DataLoader/Modelo/Train | Modelo base + curvas + métricas |
| 6 | Experimentos K-Fold + baselines | Tabla comparativa |
| 7 | Modelo final + submission | `submission.csv` + puntaje público |
| 8 | Conclusiones + Restart & Run All + README | Entrega final etiquetada |
