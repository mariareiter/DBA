# MedMind: Predicting Mental Health Risk in Spanish Medical Students

> **IE University — BDBA Capstone Project**
>
> An end-to-end machine learning pipeline that predicts depression, burnout, anxiety, and suicidal ideation among medical students using the DABE 2020 national dataset (n = 5,216), explains predictions with SHAP, and packages the results into an interactive screening tool.

---

## The Problem

Spanish medical students experience mental health problems at rates far above the general population — over 40% show depressive symptoms, roughly 37% meet criteria for academic burnout, and around 1 in 10 report suicidal ideation. Despite the scale of the problem, the country's primary support programme (PAIME) is reactive: it waits for students to seek help. Most never do. Early identification systems that could flag at-risk students before they reach crisis simply do not exist.

## What This Project Does

This project takes the largest nationally representative dataset of Spanish medical student mental health (the DABE 2020 study, covering 5,216 students across 43 universities) and does something the original authors never attempted: it builds predictive models that identify which students are most likely to be struggling, explains *why* each prediction is made, and wraps everything into a tool that could be used for proactive screening.

Three research questions drive the work:

1. **RQ1 — Descriptive:** How do depression, burnout, anxiety, and suicidal ideation vary by course year, gender, and demographic profile?
2. **RQ2 — Predictive & Explanatory:** Which machine learning models best predict each outcome, and which features drive those predictions?
3. **RQ3 — Applied:** Can these models be translated into a usable, explainable screening tool?

---

## Repository Structure

```
medmind/
│
├── notebooks/
│   ├── notebook_01_preprocessing.ipynb   — Raw data → clean datasets
│   ├── notebook_02_eda.ipynb             — Exploratory data analysis (23 figures)
│   ├── notebook_03_modelling.ipynb       — ML training, tuning, evaluation + SHAP
│   └── notebook_04_dashboard_prep.ipynb  — Export models & assets for Streamlit
│
├── data/
│   ├── dabe_clean_full.csv               — Full cleaned dataset (for EDA)
│   └── dabe_ml_features.csv              — Feature matrix + targets (for ML)
│
├── models/                               — Trained model files (.joblib)
│
├── figures/                              — All EDA and SHAP visualisations
│
├── app/
│   └── streamlit_app.py                  — Interactive screening dashboard
│
├── thesis/
│   └── methodology_trimmed.md            — Methodology chapter (for reference)
│
├── requirements.txt
└── README.md
```

---

## Dataset

The **DABE 2020** (Datos sobre el Bienestar Emocional) dataset was collected by the Spanish Conference of Medical School Deans (CNDFME) in collaboration with the Spanish Medical Students' Council (CEEM). It includes:

- **5,216 students** from **43 Spanish medical schools** (after cleaning)
- Course years 1 through 6
- Four validated psychological instruments: BDI-II (depression), MBI-SS (burnout), STAI (anxiety), JSE (empathy)
- Demographics, substance use, life events, social support, academic variables, and perceived problems

**Source:** Capdevila-Gaudens, P., García-Abajo, J.M., Flores-Funes, D., García-Barbero, M., & García-Estañ, J. (2021). Depression, anxiety, burnout and empathy among Spanish medical students. *PLOS ONE*, 16(4), e0244889.

> **Note:** The raw dataset is not included in this repository as it was obtained under a data-sharing agreement. Contact the original authors to request access.

---

## Prediction Targets

| Target | Definition | Prevalence |
|--------|-----------|------------|
| **Clinical Depression** | BDI-II ≥ 20 (moderate or severe) | ~23.4% |
| **Academic Burnout** | MBI-SS Exhaustion ≥ 2.8 AND Cynicism ≥ 2.25 | ~36.8% |
| **High Trait Anxiety** | STAI-Trait ≥ 56 (High + Very High categories) | ~22.8% |
| **Suicidal Ideation** | BDI-II Item 9 > 0 (any endorsement) | ~10.6% |

---

## Methods

### Preprocessing (Notebook 01)

Ten cleaning steps applied to the raw Excel file:

- Drop 7 empty rows and unnamed separator columns
- Cap age outliers at 17–50, impute 2 missing ages with median
- Recode all categorical variables to analysis-ready formats
- Compute composite scores for BDI-II, MBI-SS (with codification sheet verification), STAI (with reverse scoring), and Jefferson Empathy
- Parse multi-select columns (perceived problems, psychopharmaceuticals) using a custom three-format parser
- Create four binary target variables
- Median/mode imputation for remaining missingness (<1% of values)

A critical preprocessing discovery: the MBI-SS item-to-subscale mapping in the dataset does not follow the standard Schaufeli et al. (2002) numbering. The survey items were reordered, and only the codification sheet reveals the correct grouping. Using the standard numbering inflates burnout prevalence from 37% to 44%. This is documented in the thesis methodology and verified against the published paper.

### Exploratory Analysis (Notebook 02)

23 figures covering:

- Sample demographics and extended demographics
- Target prevalence and composite score distributions
- Depression and anxiety severity category breakdowns
- Year-by-year progression for all four targets (RQ1)
- Gender, sexual orientation, and work status comparisons
- Social support and academic variable analysis
- Correlation matrix, perceived problems, substance use, life events
- Dose-response relationships (life events × all 4 targets)
- Risk factor profiling (depressed vs non-depressed, SI vs non-SI)
- Comorbidity analysis with conditional probabilities
- Preclinical vs clinical training phase comparison

### Machine Learning (Notebook 03)

Four models trained on each of the four targets (16 combinations):

| Model | Rationale |
|-------|-----------|
| **Logistic Regression** | Interpretable baseline, L1/L2 regularisation |
| **Random Forest** | Ensemble of decision trees, handles non-linearity |
| **XGBoost** | State-of-the-art gradient boosting for tabular data |
| **Multilayer Perceptron** | Tests whether neural networks add value on this data size |

Pipeline:
- 80/20 stratified train/test split (consistent across all targets)
- Hyperparameter tuning via GridSearchCV with 5-fold stratified CV
- Class imbalance handling: cost-sensitive learning, SMOTE (within folds), threshold optimisation
- Evaluation: AUC-ROC, F1, precision, recall, confusion matrices
- SHAP analysis for model explainability (global and individual-level)

### Screening Tool (Streamlit App)

An interactive dashboard where a user can:
- Answer a simplified set of screening questions
- Receive personalised risk estimates for each of the four outcomes
- See which factors in their profile contribute most to their risk (SHAP waterfall)
- Access tailored recommendations and support resources, including the PAIME programme

---

## Key Findings

*Results will be populated after modelling is complete.*

---

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/medmind.git
cd medmind

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

### Requirements

- Python 3.9+
- pandas, numpy, scipy
- scikit-learn, xgboost, imbalanced-learn
- shap
- matplotlib, seaborn
- streamlit
- joblib

Full dependency list in `requirements.txt`.

### Running the Notebooks

The notebooks are designed to run in **Google Colab**. Upload the raw dataset to `/content/` and run sequentially:

1. `notebook_01` → produces `data/dabe_clean_full.csv` and `data/dabe_ml_features.csv`
2. `notebook_02` → produces 23 figures in working directory
3. `notebook_03` → produces trained models, evaluation metrics, and SHAP plots
4. `notebook_04` → exports assets for the Streamlit dashboard

---

## References

- Beck, A.T., Steer, R.A., & Brown, G.K. (1996). *Manual for the Beck Depression Inventory–II*. Psychological Corporation.
- Capdevila-Gaudens, P., et al. (2021). Depression, anxiety, burnout and empathy among Spanish medical students. *PLOS ONE*, 16(4), e0244889.
- Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. 22nd ACM SIGKDD*, 785–794.
- Galán, F., et al. (2011). Burnout, depression and suicidal ideation in medical students. *Medical Education*, 45(2), 190–197.
- Hojat, M., et al. (2001). The Jefferson Scale of Physician Empathy. *Medical Education*, 35(6), 553–562.
- Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Schaufeli, W.B., et al. (2002). Burnout and engagement in university students. *Journal of Cross-Cultural Psychology*, 33(5), 464–481.
- Spielberger, C.D., et al. (1983). *Manual for the State-Trait Anxiety Inventory*. Consulting Psychologists Press.

---

## License

This project is submitted as an academic capstone at IE University. The code is available for educational and research purposes. The DABE dataset is not redistributable — contact the original research team for access.

---

## Author

**María Reiter**
Bachelor in Data & Business Analytics
IE University, Madrid

Supervised by **Luis Galindo**
