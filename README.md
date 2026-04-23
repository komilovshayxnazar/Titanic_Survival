# Titanic Survival Prediction — MLOps Pipeline

MLflow yordamida qurilgan to'liq MLOps loyihasi. Titanic dataseti asosida yo'lovchilarning tirik qolish ehtimolini bashorat qiladi.

---

## Loyiha strukturasi

```
project/
├── configs/
│   └── config.yaml          # Barcha giperhyperparametr va sozlamalar
├── src/
│   ├── data_preprocessing.py  # Data yuklash, tozalash, masshtablash
│   ├── model_training.py      # 3 ta model trening + metrika hisoblash
│   ├── model_registry.py      # MLflow Model Registry boshqaruvi
│   └── pipeline.py            # Asosiy pipeline (ishga tushirish nuqtasi)
├── data/                      # Dataset papkasi (gitignore'd)
├── notebooks/                 # Eksperimentlar uchun Jupyter notebooks
├── .gitignore
└── README.md
```

---

## Qo'llaniladigan texnologiyalar

| Texnologiya | Maqsad |
|---|---|
| **MLflow** | Experiment tracking, model registry, artifact store |
| **scikit-learn** | ML modellar (LR, RF, GB) |
| **pandas / numpy** | Data manipulyatsiya |
| **seaborn / matplotlib** | Vizualizatsiya |
| **PyYAML** | Konfiguratsiya |

---

## Modellar va natijalar

Uchta model o'qitiladi va ROC-AUC bo'yicha eng yaxshisi avtomatik tarzda Production ga o'tkaziladi:

| Model | F1 | ROC-AUC | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.7188 | 0.8519 | 0.7989 |
| **Random Forest** | **0.7333** | **0.8525** | **0.8212** |
| Gradient Boosting | 0.7143 | 0.8203 | 0.7989 |

---

## O'rnatish

```bash
git clone https://github.com/komilovshayxnazar/Titanic_Survival.git
cd Titanic_Survival

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install mlflow scikit-learn pandas numpy seaborn matplotlib pyyaml
```

---

## Ishga tushirish

Titanic dataseti (`titanic.csv`) ni `data/` papkasiga joylashtiring, so'ng:

```bash
cd src
python3 pipeline.py
```

Pipeline quyidagilarni bajaradi:
1. Datani yuklash va tozalash
2. 3 ta modelni o'qitish (nested MLflow runs)
3. Metrikalarni solishtirish
4. Eng yaxshi modelni Model Registry ga yuklash va Production ga o'tkazish
5. Namuna inferens natijalarini chiqarish

---

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Brauzerda `http://127.0.0.1:5000` ga kiring:

- **Experiments** — har bir run ning metrika va parametrlarini ko'ring
- **Models** — ro'yxatdan o'tgan modellar va versiyalarini boshqaring
- **Artifacts** — confusion matrix, feature importance grafiklarini ko'ring

---

## MLOps arxitektura

```
[Data] → [Preprocessing] → [Training x3]
                                  ↓
                         [Metrics Comparison]
                                  ↓
                      [Best Model → Registry]
                                  ↓
                        [Production Stage]
                                  ↓
                          [Inference API]
```

---

## Konfiguratsiya

`configs/config.yaml` faylida barcha sozlamalarni o'zgartirish mumkin:

```yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: 5
    random_state: 42
```

---

## Muallif

**Shayxnazar Komilov** — [GitHub](https://github.com/komilovshayxnazar)
