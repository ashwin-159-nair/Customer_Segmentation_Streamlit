Customer Segmentation & Prediction App
This project demonstrates a complete end-to-end data science workflow, from Exploratory Data Analysis (EDA) and K-Means Clustering to deploying a real-time prediction interface using Streamlit.
The core goal is to segment a retail customer dataset based on purchasing behavior and demographics to identify key customer profiles for targeted marketing.

Repository Contents
1. Analysis_Model.ipynb -	Detailed notebook covering data cleaning, feature engineering, EDA, the Elbow Method, K-Means model training, and PCA visualization.
2. segmentation.py -	The Streamlit web application that loads the saved model and allows users to input new customer data for real-time segment prediction and visualization.
3. kmeans_model.pkl -	The saved, trained K-Means clustering model (K=6).
4. scaler.pkl -	The saved data scaling object (StandardScaler) crucial for normalizing input data before prediction.
5. customer_segmentation.csv - The original raw dataset used for training and analysis.
6. cluster_summary.csv -	Pre-calculated mean values for each cluster, used by the Streamlit app's radar chart for comparison.
7. requirements.txt -	List of all necessary Python libraries.

Key Customer Segments
The K-Means model identified 6 distinct customer segments based on standardized features:
Cluster	                                       Profile Summary	                                            Key Characteristics
  2	                                          VIP In-Store Buyers	                                Highest Income/Spending; Very Low Web Visits.
  4	                                          Traditional High Spenders	                          Oldest Segment; High Spending; Very Low Digital Engagement.
  0	                                          Balanced Engaged Shoppers	                          Moderate Income/Spending; High Web and Store Purchases.
  1	                                          Recent Budget Buyers	                              Very Low Income/Spending; Lowest Recency (most recent purchase).
  3	                                          Dormant Low Value	                                  Low Income/Spending; Highest Recency (most inactive).
  5	                                          Extreme Outlier	                                    Single data point with anomalous income; excluded from strategic use.

How to Run the Application
To run the interactive Streamlit application on your local machine:

1. Set up the Environment
First, install the required libraries using the provided requirements.txt file
pip install -r requirements.txt

2. Launch the Streamlit App
Navigate to the project directory in your terminal and execute the Streamlit command :
streamlit run segmentation.py
The app will open automatically in your web browser.
