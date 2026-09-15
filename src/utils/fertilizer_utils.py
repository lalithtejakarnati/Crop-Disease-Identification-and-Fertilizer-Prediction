import pandas as pd


# ============================================================
# FERTILIZER DATA PREPARATION
# ============================================================

def build_agronomic_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-sound agronomic features strictly from prediction-time inputs.
    Features:
    - NP_ratio, NK_ratio, PK_ratio: nutrient ratios
    - N_frac, P_frac, K_frac: proportions of each macronutrient
    - Total_NPK: sum of macronutrients
    - NP_diff, NK_diff, PK_diff: relative nutrient balance differences
    - Moist_Hum_Ratio: soil moisture to ambient humidity ratio
    - Temp_Moist: thermal-moisture interaction
    """
    df_feat = df_in.copy()
    eps = 1e-5

    # Nutrient Ratios
    df_feat["NP_ratio"] = df_feat["Nitrogen"] / (df_feat["Phosphorous"] + eps)
    df_feat["NK_ratio"] = df_feat["Nitrogen"] / (df_feat["Potassium"] + eps)
    df_feat["PK_ratio"] = df_feat["Phosphorous"] / (df_feat["Potassium"] + eps)

    # Total NPK & Fractions
    total_npk = df_feat["Nitrogen"] + df_feat["Phosphorous"] + df_feat["Potassium"]
    df_feat["N_frac"] = df_feat["Nitrogen"] / total_npk
    df_feat["P_frac"] = df_feat["Phosphorous"] / total_npk
    df_feat["K_frac"] = df_feat["Potassium"] / total_npk
    df_feat["Total_NPK"] = total_npk

    # Differences
    df_feat["NP_diff"] = df_feat["Nitrogen"] - df_feat["Phosphorous"]
    df_feat["NK_diff"] = df_feat["Nitrogen"] - df_feat["Potassium"]
    df_feat["PK_diff"] = df_feat["Phosphorous"] - df_feat["Potassium"]

    # Environmental interactions
    df_feat["Moist_Hum_Ratio"] = df_feat["Moisture"] / (df_feat["Humidity"] + eps)
    df_feat["Temp_Moist"] = df_feat["Temparature"] * df_feat["Moisture"]

    return df_feat



def prepare_fertilizer_input(
    temperature,
    humidity,
    moisture,
    soil_type,
    crop_type,
    nitrogen,
    potassium,
    phosphorous,
):
    """
    Create a DataFrame in the exact format expected
    by the trained fertilizer model.
    """

    return pd.DataFrame(
        [
            {
                "Temparature": temperature,
                "Humidity": humidity,
                "Moisture": moisture,
                "Soil Type": soil_type,
                "Crop Type": crop_type,
                "Nitrogen": nitrogen,
                "Potassium": potassium,
                "Phosphorous": phosphorous,
            }
        ]
    )



# ============================================================
# FERTILIZER PREDICTION
# ============================================================

def predict_fertilizer(model, fertilizer_input):
    """
    Generate a fertilizer prediction using the trained model.
    """

    prediction = model.predict(fertilizer_input)

    return prediction[0]
