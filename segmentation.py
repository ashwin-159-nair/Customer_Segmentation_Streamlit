import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Load model & scaler
# Ensuring the model and scaler files exist before loading
try:
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Error: The necessary model files (kmeans_model.pkl and scaler.pkl) were not found. Please ensure the analysis notebook has been run and the files are in the correct directory.")
    st.stop() # Stop the app if files are missing

st.title("Customer Segmentation App")
st.write("Enter customer details to predict customer segment.")


# Load or auto-create cluster_summary.csv
FEATURES = [
    "Age", "Income", "Total_Spending",
    "NumWebPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "Recency"
]

if os.path.exists("cluster_summary.csv") and os.path.getsize("cluster_summary.csv") > 0:
    cluster_summary = pd.read_csv("cluster_summary.csv", index_col=0)

else:
    st.warning("⚠ cluster_summary.csv not found or empty — attempting to generate automatically...")

    try:
        df = pd.read_csv("customer_segmentation.csv")
    except FileNotFoundError:
        st.error("Error: 'customer_segmentation.csv' not found. Cannot generate cluster summary.")
        st.stop()

    # Create calculated features
    df["Age"] = 2024 - df["Year_Birth"]
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )

    # FIX: Handle missing values (important!)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    # Scale
    df_scaled = scaler.transform(df[FEATURES])

    # Predict cluster
    df["Cluster"] = kmeans.predict(df_scaled)

    # Save summary
    cluster_summary = df.groupby("Cluster")[FEATURES].mean()
    cluster_summary.to_csv("cluster_summary.csv")

    st.success("✅ cluster_summary.csv generated successfully!")

# User Inputs (Streamlit UI)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending (sum of purchases)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# Predict & Display Results
cluster_descriptions = {
    0: "Older, moderate income, average spending. Engaged across both channels.",
    1: "Younger, low income, very minimal spending. Least engaged customers.",
    2: "Mid-age, very low spending and low engagement.",
    3: "Oldest segment, moderate spending, average engagement.",
    4: "High-income & very high spending — VIP / premium segment.",
    5: "Moderate income, high spending — frequent shoppers."
}

if st.button("Predict Segment"):
    # --- Prediction ---
    input_scaled = scaler.transform(input_data)
    cluster = int(kmeans.predict(input_scaled)[0])

    st.success(f"🎯 Predicted Segment: **Cluster {cluster}**")
    st.info(cluster_descriptions.get(cluster, "No description available."))

    # SAVE PREDICTION TO CSV
    input_data["PredictedCluster"] = cluster

    try:
        old = pd.read_csv("customer_predictions.csv")
        updated = pd.concat([old, input_data], ignore_index=True)
        updated.to_csv("customer_predictions.csv", index=False)
    except:
        input_data.to_csv("customer_predictions.csv", index=False)

    st.success("💾 Prediction saved to `customer_predictions.csv`")


    # RADAR CHART
    st.subheader("📊 Customer Profile vs. Cluster Average (Standardized Radar Chart)")

    features = FEATURES
    input_df_for_scaling = input_data[features]

    # Standardize (Z-score normalize) the data
    cluster_avg_df = cluster_summary.loc[[cluster], features]
    
    # Combine the customer input and cluster average for consistent scaling
    combined_data = pd.concat([input_df_for_scaling, cluster_avg_df])
    
    # Scale the combined data using the existing scaler object
    combined_scaled = scaler.transform(combined_data)

    # Separate the scaled data back out
    customer_scaled_values = combined_scaled[0, :]
    cluster_scaled_values = combined_scaled[1, :]

    # Prepare data for Matplotlib radar plot
    categories = features
    N = len(categories)

    # Calculate angles for the plot
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # The plot needs to be a complete circle, so we append the first value to the end.
    customer_values = np.concatenate((customer_scaled_values, [customer_scaled_values[0]]))
    cluster_values = np.concatenate((cluster_scaled_values, [cluster_scaled_values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    # 4. Create and display the plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Plot the Cluster Average
    ax.plot(angles, cluster_values, linewidth=2, label=f'Cluster {cluster} Average', color='orange')
    ax.fill(angles, cluster_values, color='orange', alpha=0.1)

    # Plot the Customer Profile
    ax.plot(angles, customer_values, linewidth=2, linestyle='dashed', label='Your Profile', color='blue')
    ax.fill(angles, customer_values, color='blue', alpha=0.05)

    # Set the axis labels and title
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, categories)
    
    # Set the limits to make the standardized plot look good
    # Data is centered around 0, setting limits from -2 to 3 SDs is typical.
    ax.set_ylim(min(customer_values.min(), cluster_values.min(), -2), max(customer_values.max(), cluster_values.max(), 3))
    
    # Ensure the radial axis labels (the numbers on the axis) are visible
    ax.set_rlabel_position(0)
    
    ax.legend(loc='lower left', bbox_to_anchor=(0.85, 0.95))

    ax.set_title(f"Customer Profile vs. Cluster {cluster} Average (Standardized)", size=12, y=1.1)
    st.pyplot(fig)

    st.success("✅ Standardized Radar chart displayed!")