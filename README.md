# 🌊 Digital Hydro-Consultant: AI for Rainwater Harvesting & Artificial Recharge
B.Tech Computer Science & Engineering Capstone Project

## 📌 Project Overview
The Digital Hydro-Consultant is an end-to-end Machine Learning Expert System designed to assess the viability of Rooftop Rainwater Harvesting (RTRWH) and Artificial Groundwater Recharge (AR) across the Indian subcontinent.

Instead of relying on generic geographical recommendations, this system utilizes a custom-built, high-fidelity dataset (merging Climate, Topography, and Soil properties) to output strict, physics-backed engineering diagnostics for any given coordinate in India.

## 🛠️ The Data Engineering Pipeline
A significant portion of this project involved building a proprietary INDIA_HYDROLOGY_MASTER dataset (~5,000 spatial grids covering India at 25km x 25km resolution) from disparate, raw APIs:

Climate & Supply (IMD Data): Processed 10-year trends, 15-minute extreme storm intensities (p95), and max dry-spell durations.

Geotechnical (ISRIC Soil Data): Extracted granular Sand/Clay percentages to calculate precise soil infiltration rates.

Topography (Open-Meteo API): Developed a custom rate-limiting scraper to pull 5x5 (25-point) high-resolution elevation grids, calculating macro-slope, relief, and terrain ruggedness (TRI).

## 🧠 The "Omniscient" Expert Engine
Because real-world labeled data for hydrology zones doesn't exist at this scale, an Expert Logic Engine was developed based on Central Ground Water Board (CGWB) guidelines. This deterministic engine strictly filters locations based on physics (e.g., heavily penalizing AR on >5° slopes or >30% clay soils to prevent landslides/flooding) and classifies India into 5 distinct Hydrological Zones.

## 🤖 Machine Learning Architecture
To transition from a rigid rule-based engine to a dynamic AI consultant:

Algorithm: Random Forest Classifier (n_estimators=300).

Class Imbalance Handling: Implemented SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model perfectly recognized rare but dangerous terrain profiles (like steep, highly rugged cliffs) without bias toward the dominant plains.

Target Leakage Prevention: The model was trained strictly on raw physical inputs (e.g., extreme storm loads, raw clay %, topography) while hiding all intermediate suitability scores and coordinates, forcing the AI to learn the actual hydrological cycle.

Performance: Achieved 92% overall accuracy, with a flawless 1.00 Precision/Recall on high-risk mountainous terrain (Zone 4) and 0.96 Recall on impermeable clay traps (Zone 3).

## 📂 Repository Structure
fetch_terrain_SF25.py: The custom rate-limited scraper used to extract 25-point topographical data.

recalibrate_expert.py: The deterministic CGWB logic engine used to synthetically label the master dataset.

train_ultimate_expert.py: The core ML pipeline (Train/Test Split, SMOTE, Model Training, Feature Importance extraction).

consult_expert.py: The final deployment script. Accepts user coordinates and outputs a localized Engineering & Hydrology Report.

Feature_Importance.png & Ultimate_Confusion_Matrix.png: Visual proofs of model safety and logic.

## 🚀 Quick Start (Testing the Consultant)
To consult the trained Expert System for a specific location:

Clone the repository and ensure INDIA_HYDROLOGY_FINAL_LABELED.parquet and hydro_ultimate_rf_model.pkl are in the root directory.

Install dependencies: pip install pandas numpy scikit-learn imbalanced-learn joblib

Run the deployment script:

Bash
python consult_expert.py
(By default, this will run a test diagnostic on the agricultural belt of North Rajasthan, providing tank sizing insights and structure recommendations).

## 🔮 Future Scope
UI Integration: Wrapping the consult_expert.py script into a Streamlit Web Dashboard for interactive, map-based querying.

Cost Estimation: Integrating real-time civil engineering excavation costs into the final report output based on the recommended structure sizes.
