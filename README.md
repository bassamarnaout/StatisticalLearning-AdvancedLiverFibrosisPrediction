# Statistical-Learning
Comparison of Machine Learning Approaches for Prediction of Advanced Liver Fibrosis in Chronic Hepatitis C Patients

## Business problem

Chronic hepatitis C (HCV) damages the liver over years, and the key question for treatment planning is how far the scarring (fibrosis) has progressed. Patients with advanced fibrosis (stage F3/F4) need closer monitoring and prioritised care because of their risk of cirrhosis and liver cancer.

The reference method for staging is a liver biopsy. It is invasive, costly, needs specialist staff, carries a risk of complications, and is unpleasant enough that patients may avoid it. Health systems with limited resources cannot biopsy every HCV patient.

## Value proposition

Use routine, inexpensive clinical data to estimate whether a patient is likely to have advanced fibrosis, so that:

- **Clinicians** get a decision-support signal on who most needs confirmatory testing or immediate treatment.
- **Patients** with a low predicted risk may avoid an invasive procedure, and those with a high predicted risk are flagged sooner.
- **Health systems** spend scarce biopsy and specialist capacity where it matters most.

## How the model fits in

| Element | In this project |
|---|---|
| Input | Per-patient demographics, symptoms, blood counts, liver enzymes (AST/ALT over time) and HCV viral load (RNA) from `HCV-Egy-Data.csv` (1385 patients, 28 features) |
| Output | Binary label: advanced fibrosis (histological stage F3/F4 = 1) vs. not (F1/F2 = 0) |
| Ground truth | The biopsy-based histological staging recorded in the last column of the dataset |
| Intended use | Triage / risk screening ahead of biopsy, not a replacement for clinical judgement |

`Discretization-Criteria.csv` documents how the original features were binned in the source dataset.

## What the code does

[comparing_machine_learning_methods.ipynb](comparing_machine_learning_methods.ipynb) walks through the standard scikit-learn workflow and compares several classifiers (logistic regression, SVM, decision tree, KNN, linear discriminant analysis, Gaussian naive Bayes, plus ensembles) using 10-fold cross-validation:

1. Load and summarise the data (statistics, distributions, correlations, skew)
2. Preprocess (scaling, normalisation, binarisation)
3. Feature selection (univariate, RFE, PCA, tree importance)
4. Evaluate resampling methods and metrics (accuracy, log loss, ROC AUC, confusion matrix)
5. Compare algorithms and ensembles, tune, and save a final model

`comparing_machine_learning_methods.py` is the script version, and `main.py` is a small standalone data-loading template.

## Current status and limitations

- With the default, unscaled features, the compared models reach only about 50% cross-validated accuracy on this dataset, which is roughly chance for a balanced binary target. The models are not yet useful for the use case above.
- The data comes from a single cohort in Egypt, so results may not transfer to other populations.
- Any real deployment would need external validation, calibration, and clinical and regulatory review.

## Running it

Open the notebook with a Python environment that has `pandas`, `numpy`, `matplotlib` and `scikit-learn`, then run all cells from the top. The notebook expects `HCV-Egy-Data.csv` in the same folder.
