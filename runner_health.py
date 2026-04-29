# =============================================================================
#  PROJECT:  Lifestyle, Running Habits, and Chronic Disease Risk Analysis
#  PROGRAM:  runner_health.py
#  AUTHOR:   Yuntao (Kevin) Tan
#  DATE:     March 2026
#
#  DESCRIPTION:
#    Large-scale simulation and analysis of 5 million amateur runner health
#    records using PySpark distributed computing. Examines how running habits,
#    sleep, diet, and stress relate to chronic disease risk (obesity,
#    hypertension, diabetes, CKD).
#
#    WHY PYSPARK:
#    At 5 million records, pandas would load the entire dataset into a single
#    machine's RAM and process it on one CPU core — slow and memory-intensive.
#    PySpark splits the data into partitions processed in parallel across
#    multiple CPU cores (locally) or nodes (on a cluster), making it practical
#    for datasets at the scale of KECC's 3M+ ESRD patient database.
#
#  RESEARCH QUESTIONS:
#    1. Do amateur runners with higher weekly mileage have lower chronic
#       disease risk compared to sedentary individuals?
#    2. Which lifestyle factors (sleep, diet, stress, running) are the
#       strongest predictors of obesity and CKD risk?
#    3. How does running experience level modify the diet-disease relationship?
#
#  OUTPUT:
#    - Console EDA summaries for research report
#    - Parquet output: ./runner_health_output/
#    - Model performance metrics (AUC, accuracy)
# =============================================================================

import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler, QuantileDiscretizer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array

# =============================================================================
# SECTION 0: SPARK SESSION SETUP
# =============================================================================

spark = SparkSession.builder \
    .appName("RunnerHealth_ChronicDisease_Risk") \
    .config("spark.sql.shuffle.partitions", "16") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

SEED = 20260301
N    = 5_000_000     # 5 million simulated records

print("=" * 70)
print("Lifestyle, Running Habits & Chronic Disease Risk")
print(f"Yuntao (Kevin) Tan  |  N = {N:,} simulated records  |  PySpark")
print("=" * 70)

# =============================================================================
# SECTION 1: DATA SIMULATION
# =============================================================================
#
#  WHY SIMULATE:
#  No single public dataset combines granular amateur running metrics
#  (pace, weekly mileage, race history) with clinical outcomes and lifestyle
#  factors at this scale. Simulation lets us:
#    - Control realistic correlations between variables (e.g., higher mileage
#      -> lower BMI -> lower CKD risk)
#    - Generate sufficient volume to justify distributed processing
#    - Demonstrate the full PySpark pipeline without privacy constraints
#
#  SIMULATION APPROACH:
#  numpy generates correlated variables using realistic population
#  distributions drawn from NHANES, Strava activity reports, and published
#  epidemiological literature. Outcome variables are derived from a
#  probabilistic model with added noise to reflect real-world uncertainty.
# =============================================================================

print("\n--- SECTION 1: Simulating 5M Records ---")
t0 = time.time()

rng = np.random.default_rng(SEED)

# ── Demographics ─────────────────────────────────────────────────────────────
age        = rng.normal(38, 12, N).clip(18, 80)
sex        = rng.choice([0, 1], N, p=[0.45, 0.55])   # 0=Female 1=Male
# Poverty-to-income ratio (higher = wealthier); runners skew slightly higher income
pir        = rng.normal(3.2, 1.5, N).clip(0.5, 5.0)

# ── Running habits ───────────────────────────────────────────────────────────
# Weekly runs: 0 = non-runner (30% of population), 1-6 for runners
is_runner  = rng.choice([0, 1], N, p=[0.30, 0.70])
run_days_wk    = np.where(is_runner, rng.integers(1, 7, N), 0).astype(float)
run_km_session = np.where(is_runner, rng.normal(8, 4, N).clip(1, 42), 0.0)
weekly_km      = run_days_wk * run_km_session
# Pace (min/km): faster runners tend to run more days
pace_min_km    = np.where(
    is_runner,
    rng.normal(6.0, 1.2, N).clip(3.5, 12.0) - (run_days_wk * 0.05),
    0.0
)
run_years  = np.where(is_runner, rng.integers(0, 20, N), 0).astype(float)
# Race experience: 0=none, 1=5K, 2=10K, 3=half, 4=full marathon
race_level = np.where(
    is_runner,
    np.minimum(rng.integers(0, 5, N), (run_years / 4).astype(int)),
    0
).astype(float)

# ── Sleep ────────────────────────────────────────────────────────────────────
# Runners tend to sleep slightly better; base 6.8 hr, +0.1 per run day
sleep_hrs    = (rng.normal(6.8, 1.1, N) + run_days_wk * 0.08).clip(4, 10)
sleep_quality = rng.normal(6.5, 1.8, N).clip(1, 10)  # 1-10 self-report

# ── Stress & mental health ───────────────────────────────────────────────────
# Stress score 1-10 (higher = more stress); running reduces stress
stress_score  = (rng.normal(5.5, 2.0, N) - run_days_wk * 0.15).clip(1, 10)
# Mental wellbeing score 1-10 (higher = better); correlated with sleep + running
mental_score  = (rng.normal(6.0, 1.8, N)
                 + run_days_wk * 0.10
                 + sleep_hrs * 0.05
                 - stress_score * 0.10).clip(1, 10)

# ── Diet ─────────────────────────────────────────────────────────────────────
# Runners tend to eat more calories but also more fiber
calories  = (rng.normal(2100, 500, N) + weekly_km * 30).clip(1200, 5000)
protein_g = (rng.normal(80, 25, N) + weekly_km * 0.8).clip(20, 300)
sugar_g   = rng.normal(85, 35, N).clip(10, 300)
fiber_g   = (rng.normal(17, 6, N) + run_days_wk * 0.3).clip(2, 60)
sodium_mg = rng.normal(3200, 800, N).clip(500, 7000)

# ── Body measures ────────────────────────────────────────────────────────────
# BMI: non-runners average ~28, runners ~24; reduced by weekly_km
bmi = (rng.normal(26.5, 5.0, N)
       - weekly_km * 0.04
       + age * 0.02
       - fiber_g * 0.03
       + sugar_g * 0.01).clip(15, 50)

waist_cm = (bmi * 2.8 + rng.normal(0, 5, N)).clip(55, 150)

# Systolic blood pressure
sbp = (rng.normal(118, 15, N)
       + age * 0.25
       + bmi * 0.4
       - run_days_wk * 0.8
       + stress_score * 0.5).clip(80, 200)
dbp = (sbp * 0.62 + rng.normal(0, 5, N)).clip(50, 120)

# ── Outcome variables (probabilistic, not deterministic) ────────────────────
# Obesity: BMI >= 30
obese = (bmi >= 30).astype(int)

# Hypertension: SBP >= 130 or DBP >= 80
htn = ((sbp >= 130) | (dbp >= 80)).astype(int)

# Diabetes risk score (0-1 probability, logistic model)
dm_logit = (-4.5
            + bmi        * 0.12
            + age        * 0.03
            + sugar_g    * 0.008
            - fiber_g    * 0.04
            - weekly_km  * 0.012
            + stress_score * 0.05
            - sleep_hrs  * 0.08
            + rng.normal(0, 0.3, N))
dm_prob  = 1 / (1 + np.exp(-dm_logit))
diabetes_risk = (dm_prob > 0.35).astype(int)

# CKD risk (chronic kidney disease) — primary outcome of interest
# Higher risk with: older age, high BMI, hypertension, diabetes, high sodium
ckd_logit = (-5.0
             + age        * 0.04
             + bmi        * 0.08
             + htn        * 0.90
             + diabetes_risk * 1.10
             + sodium_mg  * 0.0003
             - weekly_km  * 0.008
             - sleep_hrs  * 0.06
             + stress_score * 0.04
             + rng.normal(0, 0.4, N))
ckd_prob  = 1 / (1 + np.exp(-ckd_logit))
ckd_risk  = (ckd_prob > 0.30).astype(int)

print(f"  Data generation: {time.time()-t0:.1f}s")

# ── Build Spark DataFrame from numpy arrays ──────────────────────────────────
t1 = time.time()

schema = StructType([
    StructField("age",           DoubleType()),
    StructField("sex",           IntegerType()),
    StructField("pir",           DoubleType()),
    StructField("is_runner",     IntegerType()),
    StructField("run_days_wk",   DoubleType()),
    StructField("run_km_session",DoubleType()),
    StructField("weekly_km",     DoubleType()),
    StructField("pace_min_km",   DoubleType()),
    StructField("run_years",     DoubleType()),
    StructField("race_level",    DoubleType()),
    StructField("sleep_hrs",     DoubleType()),
    StructField("sleep_quality", DoubleType()),
    StructField("stress_score",  DoubleType()),
    StructField("mental_score",  DoubleType()),
    StructField("calories",      DoubleType()),
    StructField("protein_g",     DoubleType()),
    StructField("sugar_g",       DoubleType()),
    StructField("fiber_g",       DoubleType()),
    StructField("sodium_mg",     DoubleType()),
    StructField("bmi",           DoubleType()),
    StructField("waist_cm",      DoubleType()),
    StructField("sbp",           DoubleType()),
    StructField("dbp",           DoubleType()),
    StructField("obese",         IntegerType()),
    StructField("htn",           IntegerType()),
    StructField("diabetes_risk", IntegerType()),
    StructField("ckd_risk",      IntegerType()),
])

pdf = pd.DataFrame({
    "age": age, "sex": sex.astype(int), "pir": pir,
    "is_runner": is_runner.astype(int),
    "run_days_wk": run_days_wk, "run_km_session": run_km_session,
    "weekly_km": weekly_km, "pace_min_km": pace_min_km,
    "run_years": run_years, "race_level": race_level,
    "sleep_hrs": sleep_hrs, "sleep_quality": sleep_quality,
    "stress_score": stress_score, "mental_score": mental_score,
    "calories": calories, "protein_g": protein_g,
    "sugar_g": sugar_g, "fiber_g": fiber_g, "sodium_mg": sodium_mg,
    "bmi": bmi, "waist_cm": waist_cm,
    "sbp": sbp, "dbp": dbp,
    "obese": obese.astype(int), "htn": htn.astype(int),
    "diabetes_risk": diabetes_risk.astype(int),
    "ckd_risk": ckd_risk.astype(int),
})

df = spark.createDataFrame(pdf, schema=schema)
df.cache()
print(f"  Spark DataFrame created: {df.count():,} rows | {time.time()-t1:.1f}s")

# =============================================================================
# SECTION 2: DATA CLEANING & FEATURE ENGINEERING
# =============================================================================

print("\n--- SECTION 2: Data Cleaning & Feature Engineering ---")

df.unpersist()
df = (df
    # Running experience category
    .withColumn("runner_cat",
        F.when(F.col("weekly_km") == 0,    "Sedentary")
         .when(F.col("weekly_km") < 20,    "Low (<20km/wk)")
         .when(F.col("weekly_km") < 40,    "Moderate (20-40km/wk)")
         .otherwise(                        "High (40+km/wk)"))

    # BMI category (WHO)
    .withColumn("bmi_cat",
        F.when(F.col("bmi") < 18.5, "Underweight")
         .when(F.col("bmi") < 25.0, "Normal")
         .when(F.col("bmi") < 30.0, "Overweight")
         .otherwise("Obese"))

    # Sleep adequacy flag (CDC: 7+ hours recommended)
    .withColumn("flag_short_sleep",  (F.col("sleep_hrs") < 7).cast(IntegerType()))
    .withColumn("flag_high_stress",  (F.col("stress_score") >= 7).cast(IntegerType()))
    .withColumn("flag_high_sugar",   (F.col("sugar_g") > 100).cast(IntegerType()))
    .withColumn("flag_low_fiber",    (F.col("fiber_g") < 15).cast(IntegerType()))

    # Composite lifestyle score (0-10; higher = healthier)
    # Components: running, sleep, diet quality, low stress
    .withColumn("lifestyle_score",
        F.least(F.lit(10.0),
            F.greatest(F.lit(0.0),
                (F.col("weekly_km") / 20.0)          # running component
                + (F.col("sleep_hrs") - 5.0)          # sleep component
                + (F.col("fiber_g") / 10.0)           # diet component
                - (F.col("stress_score") / 5.0)       # stress penalty
            )))

    # Log-transform skewed variables for modeling
    .withColumn("log_weekly_km", F.log1p(F.col("weekly_km")))
    .withColumn("log_calories",  F.log(F.col("calories")))
    .withColumn("log_sodium",    F.log(F.col("sodium_mg")))
)

df.cache()
print(f"  Feature engineering complete. N = {df.count():,}")

# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n--- SECTION 3: Exploratory Data Analysis ---")
print("(Save these numbers for the Research Report)\n")

# ── 3.1 Sample overview ──────────────────────────────────────────────────────
print("── 3.1 Sample Overview ──")
df.select(
    F.count("*").alias("N"),
    F.round(F.mean("age"), 1).alias("mean_age"),
    F.round(F.mean("bmi"), 1).alias("mean_bmi"),
    F.round(F.mean("weekly_km"), 1).alias("mean_weekly_km"),
    F.round(F.mean("sleep_hrs"), 1).alias("mean_sleep"),
    F.round(F.mean("stress_score"), 1).alias("mean_stress"),
).show()

# Runner distribution
print("── Runner Category Distribution ──")
total_n = df.count()
df.groupBy("runner_cat").agg(
    F.count("*").alias("N"),
    F.round(F.count("*") / total_n * 100, 1).alias("Pct")
).orderBy("N", ascending=False).show()

# ── 3.2 Disease prevalence ───────────────────────────────────────────────────
print("── 3.2 Disease Prevalence ──")
df.select(
    F.round(F.mean("obese")         * 100, 1).alias("Obesity_%"),
    F.round(F.mean("htn")           * 100, 1).alias("Hypertension_%"),
    F.round(F.mean("diabetes_risk") * 100, 1).alias("Diabetes_Risk_%"),
    F.round(F.mean("ckd_risk")      * 100, 1).alias("CKD_Risk_%"),
).show()

# ── 3.3 Key outcomes by runner category ─────────────────────────────────────
print("── 3.3 Health Outcomes by Running Level ──")
df.groupBy("runner_cat").agg(
    F.count("*").alias("N"),
    F.round(F.mean("bmi"),          1).alias("Mean_BMI"),
    F.round(F.mean("sbp"),          1).alias("Mean_SBP"),
    F.round(F.mean("sleep_hrs"),    1).alias("Mean_Sleep_Hrs"),
    F.round(F.mean("stress_score"), 1).alias("Mean_Stress"),
    F.round(F.mean("obese")      * 100, 1).alias("Obesity_%"),
    F.round(F.mean("ckd_risk")   * 100, 1).alias("CKD_Risk_%"),
    F.round(F.mean("diabetes_risk") * 100, 1).alias("Diabetes_%"),
).orderBy("runner_cat").show(truncate=False)

# ── 3.4 Sleep and stress relationship with disease ───────────────────────────
print("── 3.4 CKD Risk by Sleep Adequacy and Stress Level ──")
df.groupBy("flag_short_sleep", "flag_high_stress").agg(
    F.count("*").alias("N"),
    F.round(F.mean("ckd_risk")      * 100, 1).alias("CKD_Risk_%"),
    F.round(F.mean("diabetes_risk") * 100, 1).alias("Diabetes_%"),
    F.round(F.mean("obese")         * 100, 1).alias("Obesity_%"),
).orderBy("flag_short_sleep", "flag_high_stress").show()

# ── 3.5 Diet quality by runner category ─────────────────────────────────────
print("── 3.5 Diet Patterns by Running Level ──")
df.groupBy("runner_cat").agg(
    F.round(F.mean("calories"),  0).alias("Mean_Calories"),
    F.round(F.mean("protein_g"), 1).alias("Mean_Protein_g"),
    F.round(F.mean("sugar_g"),   1).alias("Mean_Sugar_g"),
    F.round(F.mean("fiber_g"),   1).alias("Mean_Fiber_g"),
    F.round(F.mean("sodium_mg"), 0).alias("Mean_Sodium_mg"),
).orderBy("runner_cat").show(truncate=False)

# ── 3.6 Correlation: weekly_km vs key health metrics ────────────────────────
print("── 3.6 Mean Health Metrics by Weekly Mileage Decile ──")
disc = QuantileDiscretizer(numBuckets=5, inputCol="weekly_km",
                           outputCol="km_bucket", handleInvalid="skip")
df_bucketed = disc.fit(df).transform(df)
df_bucketed.groupBy("km_bucket").agg(
    F.round(F.mean("weekly_km"),    1).alias("Avg_km_wk"),
    F.round(F.mean("bmi"),          1).alias("Mean_BMI"),
    F.round(F.mean("sbp"),          1).alias("Mean_SBP"),
    F.round(F.mean("ckd_risk")   * 100, 1).alias("CKD_%"),
    F.round(F.mean("sleep_hrs"),    1).alias("Sleep_Hrs"),
    F.round(F.mean("stress_score"), 1).alias("Stress"),
    F.count("*").alias("N"),
).orderBy("km_bucket").show()

# =============================================================================
# SECTION 4: ML MODELS — PREDICTING CKD RISK
# =============================================================================

print("\n--- SECTION 4: Machine Learning — CKD Risk Prediction ---")

# ── 4.1 Prepare features ────────────────────────────────────────────────────
feature_cols = [
    # Running
    "log_weekly_km", "run_days_wk", "pace_min_km", "run_years", "race_level",
    # Sleep & mental health
    "sleep_hrs", "sleep_quality", "stress_score", "mental_score",
    "flag_short_sleep", "flag_high_stress",
    # Diet
    "log_calories", "protein_g", "sugar_g", "fiber_g", "log_sodium",
    "flag_high_sugar", "flag_low_fiber",
    # Body & demographics
    "bmi", "waist_cm", "sbp", "dbp", "age", "sex", "pir",
    # Composite
    "lifestyle_score",
    # Comorbidities
    "obese", "htn", "diabetes_risk",
]

assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip")
scaler = StandardScaler(
    inputCol="features_raw", outputCol="features",
    withMean=True, withStd=True)

# ── 4.2 Train/test split (80/20) ────────────────────────────────────────────
train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)
print(f"  Train N = {train_df.count():,} | Test N = {test_df.count():,}")

# ── 4.3 Model A: Logistic Regression ────────────────────────────────────────
print("\n  Model A: Logistic Regression")
lr = LogisticRegression(
    featuresCol="features", labelCol="ckd_risk",
    maxIter=100, regParam=0.01, elasticNetParam=0.1)

pipeline_lr = Pipeline(stages=[assembler, scaler, lr])
model_lr    = pipeline_lr.fit(train_df)
preds_lr    = model_lr.transform(test_df)

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="ckd_risk", metricName="areaUnderROC")
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="ckd_risk", predictionCol="prediction", metricName="accuracy")

auc_lr = evaluator_auc.evaluate(preds_lr)
acc_lr = evaluator_acc.evaluate(preds_lr)
print(f"    AUC = {auc_lr:.3f}  |  Accuracy = {acc_lr:.3f}")

# ── 4.4 Model B: Random Forest ──────────────────────────────────────────────
print("\n  Model B: Random Forest")
rf = RandomForestClassifier(
    featuresCol="features", labelCol="ckd_risk",
    numTrees=50, maxDepth=6, seed=SEED)

pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
model_rf    = pipeline_rf.fit(train_df)
preds_rf    = model_rf.transform(test_df)

auc_rf = evaluator_auc.evaluate(preds_rf)
acc_rf = evaluator_acc.evaluate(preds_rf)
print(f"    AUC = {auc_rf:.3f}  |  Accuracy = {acc_rf:.3f}")

# ── 4.5 Feature importance (Random Forest) ──────────────────────────────────
print("\n  Top 10 Feature Importances (Random Forest):")
rf_model = model_rf.stages[-1]
importances = rf_model.featureImportances.toArray()
feat_imp = sorted(zip(feature_cols, importances),
                  key=lambda x: x[1], reverse=True)[:10]
for feat, imp in feat_imp:
    print(f"    {feat:<25} {imp:.4f}  {'█' * int(imp * 200)}")

# ── 4.6 Model comparison summary ────────────────────────────────────────────
print("\n── Model Comparison ──")
print(f"  {'Model':<25} {'AUC':>8} {'Accuracy':>10}")
print(f"  {'-'*45}")
print(f"  {'Logistic Regression':<25} {auc_lr:>8.3f} {acc_lr:>10.3f}")
print(f"  {'Random Forest':<25} {auc_rf:>8.3f} {acc_rf:>10.3f}")

# =============================================================================
# SECTION 5: CALIBRATION & RISK STRATIFICATION
# =============================================================================

print("\n--- SECTION 5: Risk Stratification ---")

# Add predicted probability column
preds_final = (
    model_rf.transform(test_df)
    .withColumn("ckd_pred_prob", vector_to_array(F.col("probability")).getItem(1))
)

# Risk tiers
preds_stratified = preds_final.withColumn("risk_tier",
    F.when(F.col("ckd_pred_prob") < 0.20, "Low")
     .when(F.col("ckd_pred_prob") < 0.45, "Moderate")
     .otherwise("High"))

print("CKD Risk Tier Distribution:")
preds_stratified.groupBy("risk_tier").agg(
    F.count("*").alias("N"),
    F.round(F.mean("ckd_risk")      * 100, 1).alias("Actual_CKD_%"),
    F.round(F.mean("ckd_pred_prob") * 100, 1).alias("Mean_Pred_Prob_%"),
    F.round(F.mean("weekly_km"),    1).alias("Avg_Weekly_km"),
    F.round(F.mean("bmi"),          1).alias("Avg_BMI"),
    F.round(F.mean("sleep_hrs"),    1).alias("Avg_Sleep"),
    F.round(F.mean("stress_score"), 1).alias("Avg_Stress"),
).orderBy("risk_tier").show(truncate=False)

# =============================================================================
# SECTION 6: SUBGROUP ANALYSIS — RUNNERS vs NON-RUNNERS
# =============================================================================

print("\n--- SECTION 6: Runner vs Non-Runner Deep Dive ---")

# Among high mileage runners only
high_runners = df.filter(F.col("weekly_km") >= 40)
sedentary    = df.filter(F.col("weekly_km") == 0)

print(f"  High-mileage runners (40+ km/wk): N = {high_runners.count():,}")
print(f"  Sedentary (0 km/wk):               N = {sedentary.count():,}")

for label, subdf in [("High Runners", high_runners), ("Sedentary", sedentary)]:
    stats = subdf.select(
        F.round(F.mean("bmi"),          1).alias("BMI"),
        F.round(F.mean("sleep_hrs"),    1).alias("Sleep"),
        F.round(F.mean("stress_score"), 1).alias("Stress"),
        F.round(F.mean("ckd_risk")   * 100, 1).alias("CKD_%"),
        F.round(F.mean("obese")      * 100, 1).alias("Obese_%"),
        F.round(F.mean("diabetes_risk") * 100, 1).alias("Diabetes_%"),
    ).collect()[0]
    print(f"\n  [{label}]")
    for k, v in stats.asDict().items():
        print(f"    {k:<15} {v}")

# =============================================================================
# SECTION 7: OUTPUT & EXPORT
# =============================================================================

print("\n--- SECTION 7: Output & Export ---")

# Save full scored dataset as Parquet (efficient for large-scale storage)
output_path = "./runner_health_output/"
preds_final.select(
    "age", "sex", "bmi", "bmi_cat", "runner_cat",
    "weekly_km", "sleep_hrs", "stress_score", "mental_score",
    "lifestyle_score", "obese", "htn", "diabetes_risk", "ckd_risk",
    "ckd_pred_prob"
).write.mode("overwrite").parquet(output_path)

print(f"  Scored dataset saved to: {output_path}")
print(f"  Format: Parquet (columnar, compressed — efficient for large datasets)")

# Summary stats for research report
print("\n── FINAL SUMMARY FOR RESEARCH REPORT ──")
print(f"  Total records analyzed : {N:,}")
print(f"  Runner prevalence      : {df.filter(F.col('is_runner')==1).count()/N*100:.1f}%")
print(f"  Best model (RF) AUC    : {auc_rf:.3f}")
print(f"  Best model (RF) Acc    : {acc_rf:.3f}")
print(f"  Top predictor          : {feat_imp[0][0]}")

print("\n" + "=" * 70)
print("Pipeline COMPLETE.")
print("=" * 70)

spark.stop()