import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# PHASE 1: THE OMNISCIENT LOGIC ENGINE (LABELING)
# ==========================================
print("🌍 Loading Master Dataset...")
# Make sure this points to your merged parquet file!
df = pd.read_parquet("INDIA_HYDROLOGY_MASTER.parquet")

print("🧑‍🔬 Calculating Advanced Ground Truth Labels...")

# --- 1. Advanced AR (Recharge) Suitability ---
# We use the infiltration rate here to label the data, but we will HIDE it from the AI later.
inf_score = np.clip((df["infiltration_rate_mm_hr"] / 25.0) * 100, 0, 100)
topo_score = np.clip(100 - (df["SLOPE_DEG"] * 15) - (df["RUGGEDNESS_TRI"] / 10), 0, 100)
rain_score = np.clip((df["recent_10yr_avg_mm"] / 1200.0) * 100, 0, 100)
disaster_penalty = np.clip(
    (df["cv_reliability"] * 30) + (df["p95_daily_mm"] / 5), 0, 40
)

df["AR_ADV_SCORE"] = np.clip(
    (inf_score * 0.40) + (topo_score * 0.40) + (rain_score * 0.20) - disaster_penalty,
    0,
    100,
)

# --- 2. Advanced RWH (Storage) Suitability ---
drought_risk = np.clip(
    (df["avg_max_dry_days"] + (df["trend_dry_days_per_year"] * 15)) / 150.0 * 100,
    0,
    100,
)
burst_bonus = np.clip(df["design_15min_filter_intensity_mm"] / 30.0 * 100, 0, 100)

df["RWH_ADV_SCORE"] = np.clip(
    (rain_score * 0.40) + (drought_risk * 0.40) + (burst_bonus * 0.20), 0, 100
)


# --- 3. Expert Zone Assignment ---
def assign_expert_zone_v3(row):
    ar = row["AR_ADV_SCORE"]
    rwh = row["RWH_ADV_SCORE"]
    slope = row["SLOPE_DEG"]
    inf = row["infiltration_rate_mm_hr"]
    rugged = row["RUGGEDNESS_TRI"]

    if slope >= 5.0 or rugged > 250:
        return "Zone 4: Storage Priority (Hilly/Rugged Terrain - AR Unsafe)"
    elif ar >= 60 and rwh >= 60:
        return "Zone 1: Dual System (High AR + High RWH)"
    elif ar >= 60 and rwh < 60:
        return "Zone 2: Primary Recharge (Favorable Soil/Terrain)"
    elif ar < 45 and inf < 15:
        return "Zone 3: Surface Storage Priority (Impermeable Soil)"
    else:
        return "Zone 5: Mixed/Moderate Potential"


df["EXPERT_ZONE"] = df.apply(assign_expert_zone_v3, axis=1)
print("\n📊 --- NEW LABEL DISTRIBUTION ---")
print(df["EXPERT_ZONE"].value_counts())


# ==========================================
# PHASE 2: ML MODEL TRAINING (NO TARGET LEAKAGE)
# ==========================================
print("\n🤖 Assembling Machine Learning Features (Strictly Raw Physics)...")

# Notice what is MISSING: No coordinates, no intermediate scores, no infiltration rates, no text labels.
features = [
    # Climate & Supply
    "annual_avg_mm",
    "recent_10yr_avg_mm",
    "cv_reliability",
    "avg_max_dry_days",
    "p95_daily_mm",
    "peak_daily_mm",
    "trend_dry_days_per_year",
    "trend_rainy_days_per_year",
    "trend_sdii_intensity",
    "trend_p95_intensity",
    "trend_peak_intensity",
    "design_15min_filter_intensity_mm",
    "design_15min_overflow_intensity_mm",
    # Soil (The Sponge)
    "avg_sand_pct",
    "avg_clay_pct",
    # Topography (Gravity & Flow)
    "ELEVATION_MEAN",
    "RELIEF_M",
    "RUGGEDNESS_TRI",
    "SLOPE_DEG",
    "CURVATURE",
]

X = df[features]
y = df["EXPERT_ZONE"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# --- SMOTE Oversampling ---
print("🧬 Applying SMOTE to balance rare terrain/soil classes...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- Train the RF Model ---
print("⚙️ Training the Ultimate Random Forest...")
rf_expert = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1, max_depth=20
)
rf_expert.fit(X_train_smote, y_train_smote)

# --- Predictions & Grading ---
y_pred = rf_expert.predict(X_test)

print("\n📊 --- CAPSTONE CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# ==========================================
# PHASE 3: VISUALIZATION & EXPORT
# ==========================================
print("\n🗺️ Generating Confusion Matrix...")
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred, labels=rf_expert.classes_)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=rf_expert.classes_,
    yticklabels=rf_expert.classes_,
    linewidths=0.5,
    cbar_kws={"shrink": 0.75},
)
plt.title("Ultimate Expert System - Confusion Matrix", pad=20, fontsize=16)
plt.ylabel("True Zone (Expert Ground Truth)", weight="bold")
plt.xlabel("Predicted Zone (AI Output)", weight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("Ultimate_Confusion_Matrix.png", dpi=300)
print("📸 Confusion matrix saved to 'Ultimate_Confusion_Matrix.png'")

print("\n🔍 --- WHAT DRIVES THE DECISION? (Feature Importance) ---")
importances = rf_expert.feature_importances_
feature_imp_df = pd.DataFrame(
    {"Feature": features, "Importance (%)": importances * 100}
)
feature_imp_df = feature_imp_df.sort_values(by="Importance (%)", ascending=False)

for index, row in feature_imp_df.iterrows():
    print(f"{row['Feature']:>35}: {row['Importance (%)']:.2f}%")

# Plot Feature Importances
plt.figure(figsize=(10, 8))
sns.barplot(
    x="Importance (%)", y="Feature", data=feature_imp_df.head(15), palette="viridis"
)
plt.title("Top 15 Most Important Features for Hydrological Zoning")
plt.tight_layout()
plt.savefig("Feature_Importance.png", dpi=300)
print("📸 Feature Importance chart saved to 'Feature_Importance.png'")

# --- Save Final Assets ---
df.to_parquet("INDIA_HYDROLOGY_FINAL_LABELED.parquet", index=False)
joblib.dump(rf_expert, "hydro_ultimate_rf_model.pkl")
print("\n💾 Pipeline Complete. Dataset & Model Saved!")
