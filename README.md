# Lifestyle, Running Habits, and Chronic Disease Risk

A large-scale distributed analysis of **5 million simulated adult health records** using Apache PySpark, examining how running habits, sleep, diet, and stress relate to chronic disease risk (obesity, hypertension, diabetes, CKD).

**[View Research Report](./index.html)**

---

## Overview

This project demonstrates a production-grade PySpark pipeline applied to a healthcare research question. At 5 million records, single-machine pandas processing becomes a bottleneck — PySpark partitions data across multiple CPU cores (locally) or cluster nodes (in production), making it practical for datasets at the scale of large healthcare administrative databases.

### Research Questions

1. Do amateur runners at varying mileage levels show lower chronic disease prevalence than sedentary adults, and is there evidence of a dose-response relationship?
2. Which lifestyle factors are the strongest independent predictors of CKD risk after controlling for clinical covariates?
3. How accurately can machine learning models classify CKD risk using lifestyle and demographic variables?

---

## Key Findings

| Comparison | CKD Risk | Diabetes Risk | Obesity |
|---|---|---|---|
| Sedentary | 91.3% | 63.6% | 31.7% |
| High runners (≥40 km/wk) | 65.3% | 19.7% | 17.6% |
| **Difference** | **−26.0 pp** | **−43.9 pp** | **−14.1 pp** |

- **Dose-response confirmed**: CKD risk declines monotonically from 90.6% (Q1, avg 1.2 km/wk) to 66.3% (Q4, avg 54.9 km/wk)
- **Sleep × stress additive effect**: Short sleep + high stress → 89.8% CKD risk vs 78.9% with adequate sleep + low stress (+11 pp)
- **Sugar intake is uniform** across all runner categories (~85 g/day) — a modifiable risk factor decoupled from exercise level

### Model Performance

| Model | AUC | Accuracy |
|---|---|---|
| Logistic Regression | 0.971 | 92.8% |
| Random Forest | 0.952 | 90.5% |

Top feature importances (Random Forest): diabetes risk (0.2186), hypertension (0.1816), age (0.1194), systolic BP (0.1140), BMI (0.1024). Running-specific features (`log_weekly_km`, `lifestyle_score`) retain independent predictive value at ranks 9–10 after clinical covariate adjustment.

---

## Project Structure

```
runner-health/
├── runner_health.py                   # Main PySpark analysis pipeline
├── runner_health_research_report.html # Full research report with visualizations
└── README.md
```

---

## Pipeline Structure

```
Section 1 — Data Simulation        5M records, numpy, correlated variable structure
Section 2 — Feature Engineering    Runner categories, lifestyle score, log-transforms
Section 3 — EDA                    Group comparisons, mileage quintiles, sleep × stress
Section 4 — ML Models              Logistic regression + Random Forest, AUC evaluation
Section 5 — Risk Stratification    Low / Moderate / High CKD risk tiers
Section 6 — Subgroup Analysis      High runners vs. sedentary deep dive
Section 7 — Output                 Parquet export
```

---

## Data

Data are fully **simulated** using `numpy` with parameter distributions informed by:

- [NHANES 2017–2020 Pre-Pandemic Data](https://wwwn.cdc.gov/nchs/nhanes/Search/DataPage.aspx?Component=Examination&Cycle=2017-2020) — body measures, activity, diet distributions
- [Strava Global Activity Report 2023](https://stories.strava.com/articles/strava-year-in-sport-trend-report-insights-on-the-world-of-exercise) — amateur running behavior
- Published epidemiological literature on lifestyle-CKD associations

Variable relationships are encoded with realistic correlated structure plus Gaussian noise. Outcomes are derived from probabilistic logistic models. **Findings should not be interpreted as causal evidence from real populations.**

> Note: Overall CKD prevalence (82.6%) is substantially higher than the real-world rate (~15%), reflecting simulation parameters calibrated for class balance rather than epidemiological fidelity.

---

## Setup & Usage

### Requirements

- Python 3.9+
- Java 17+ (required by PySpark)
- Apache PySpark 4.x

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/runner-health.git
cd runner-health

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pyspark numpy pandas

# Install Java (macOS)
brew install openjdk@17
export JAVA_HOME=$(brew --prefix openjdk@17)
export PATH=$JAVA_HOME/bin:$PATH
```

### Run

```bash
python3 runner_health.py
```

Runtime: ~5–8 minutes on a modern laptop (Apple Silicon or equivalent).  
Output: scored dataset saved to `./runner_health_output/` in Parquet format.

### View the Report

Open `index.html` in any browser — no server required.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Apache PySpark 4.1 | Distributed data processing |
| `pyspark.ml` | ML pipelines — LogisticRegression, RandomForestClassifier, StandardScaler |
| numpy | Data simulation |
| pandas | DataFrame bridge for Spark ingestion |
| Parquet | Columnar output storage |
| Chart.js | Interactive visualizations in research report |

---

## Author

**Yuntao (Kevin) Tan**  
[tyuntao@umich.edu](mailto:tyuntao@umich.edu) · [Portfolio](https://bit.ly/47wvua3)

---

## References

1. Centers for Disease Control and Prevention (CDC). *Chronic Kidney Disease in the United States, 2023.* Atlanta, GA: US Department of Health and Human Services, CDC; 2023.
2. Jha V, Garcia-Garcia G, Iseki K, et al. Chronic kidney disease: global dimension and perspectives. *Lancet.* 2013;382(9888):260–272.
3. Thomas G, Sehgal AR, Kashyap SR, et al. Metabolic syndrome and kidney disease: a systematic review and meta-analysis. *Clin J Am Soc Nephrol.* 2011;6(10):2364–2373.
4. Howden EJ, Leano R, Petchey W, et al. Effects of exercise and lifestyle intervention on cardiovascular function in CKD. *Clin J Am Soc Nephrol.* 2013;8(9):1494–1501.
5. Ricardo AC, Anderson CA, Yang W, et al. Healthy lifestyle and risk of kidney disease progression, atherosclerotic events, and death in CKD. *Am J Kidney Dis.* 2015;65(3):412–424.
6. Strava Inc. *2023 Year in Sport: Strava Global Activity Report.* San Francisco, CA: Strava; 2024.
7. Leehey DJ, Collins E, Kramer HJ, et al. Structured exercise in obese diabetic patients with chronic kidney disease: a randomized controlled trial. *Am J Nephrol.* 2016;44(1):54–62.
8. National Center for Health Statistics. *National Health and Nutrition Examination Survey 2017–March 2020 Pre-Pandemic Data Files.* Hyattsville, MD: CDC; 2021.
9. Webster AC, Nagler EV, Morton RL, Masson P. Chronic kidney disease. *Lancet.* 2017;389(10075):1238–1252.
10. Centers for Medicare & Medicaid Services (CMS). *2022 CMS Research, Statistics, Data and Systems.* Baltimore, MD: CMS; 2022.
11. Watson NF, Badr MS, Belenky G, et al. Recommended amount of sleep for a healthy adult: a joint consensus statement of the American Academy of Sleep Medicine and Sleep Research Society. *J Clin Sleep Med.* 2015;11(6):591–592.
12. Wasserstein RL, Schirm AL, Lazar NA. Moving to a world beyond "p < 0.05." *Am Stat.* 2019;73(sup1):1–19.
13. Spiegel K, Tasali E, Leproult R, Van Cauter E. Effects of poor and short sleep on glucose metabolism and obesity risk. *Nat Rev Endocrinol.* 2009;5(5):253–261.
14. Cohen S, Janicki-Deverts D, Miller GE. Psychological stress and disease. *JAMA.* 2007;298(14):1685–1687.
15. World Health Organization. *Guideline: Sugars Intake for Adults and Children.* Geneva: WHO; 2015.
