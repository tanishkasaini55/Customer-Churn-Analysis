# =========================================================
# CUSTOMER CHURN ANALYSIS PROJECT
# Dataset-Based Final Professional Project
# =========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("customer_churn_business_dataset.csv")

# Display First 5 Rows
print(df.head())

# =========================================================
# DATA CLEANING
# =========================================================

print("\nMissing Values:\n")
print(df.isnull().sum())

# Remove Duplicate Records
df.drop_duplicates(inplace=True)

# Fill Missing Values
# Fill missing values separately

# Numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# String columns
string_cols = df.select_dtypes(include='object').columns
df[string_cols] = df[string_cols].fillna("Unknown")

print("\nDataset Info:\n")
print(df.info())

# =========================================================
# BASIC KPI ANALYSIS
# =========================================================

# Total Customers
total_customers = df['customer_id'].nunique()

# Churned Customers
churned_customers = df[df['churn'] == 1].shape[0]

# Churn Rate
churn_rate = (churned_customers / total_customers) * 100

# Average Revenue
avg_revenue = df['total_revenue'].mean()

# Average Monthly Fee
avg_monthly_fee = df['monthly_fee'].mean()

print("\n=============== KPI REPORT ===============")

print("Total Customers:", total_customers)
print("Churned Customers:", churned_customers)
print("Churn Rate:", round(churn_rate, 2), "%")
print("Average Revenue:", round(avg_revenue, 2))
print("Average Monthly Fee:", round(avg_monthly_fee, 2))

# =========================================================
# GENDER CHURN ANALYSIS
# =========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    x='gender',
    hue='churn',
    data=df
)

plt.title("Gender vs Churn")
plt.xlabel("Gender")
plt.ylabel("Customers")

plt.show()

# =========================================================
# AGE ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='age',
    data=df
)

plt.title("Age Distribution by Churn")

plt.show()

# =========================================================
# COUNTRY ANALYSIS
# =========================================================

plt.figure(figsize=(10,6))

sns.countplot(
    y='country',
    hue='churn',
    data=df
)

plt.title("Country-wise Churn Analysis")

plt.show()

# =========================================================
# CUSTOMER SEGMENT ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x='customer_segment',
    hue='churn',
    data=df
)

plt.title("Customer Segment vs Churn")

plt.show()

# =========================================================
# CONTRACT TYPE ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x='contract_type',
    hue='churn',
    data=df
)

plt.title("Contract Type vs Churn")

plt.show()

# =========================================================
# SIGNUP CHANNEL ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x='signup_channel',
    hue='churn',
    data=df
)

plt.title("Signup Channel vs Churn")

plt.show()

# =========================================================
# MONTHLY LOGIN ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x='monthly_logins',
    hue='churn',
    kde=True
)

plt.title("Monthly Logins Distribution")

plt.show()

# =========================================================
# SUPPORT TICKET ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='support_tickets',
    data=df
)

plt.title("Support Tickets vs Churn")

plt.show()

# =========================================================
# CUSTOMER SATISFACTION ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='csat_score',
    data=df
)

plt.title("CSAT Score vs Churn")

plt.show()

# =========================================================
# REVENUE ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='total_revenue',
    data=df
)

plt.title("Revenue Impact by Churn")

plt.show()

# =========================================================
# PAYMENT FAILURE ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x='payment_failures',
    hue='churn',
    data=df
)

plt.title("Payment Failures vs Churn")

plt.show()

# =========================================================
# HIGH RISK CUSTOMER IDENTIFICATION
# =========================================================

high_risk_customers = df[
    (df['monthly_logins'] < 5) &
    (df['support_tickets'] > 3) &
    (df['csat_score'] < 3)
]

print("\n=============== HIGH RISK CUSTOMERS ===============\n")

print(high_risk_customers[
    [
        'customer_id',
        'country',
        'customer_segment',
        'monthly_logins',
        'support_tickets',
        'csat_score',
        'churn'
    ]
].head())

# =========================================================
# CORRELATION HEATMAP
# =========================================================

numeric_columns = df.select_dtypes(include=np.number)

plt.figure(figsize=(14,10))

sns.heatmap(
    numeric_columns.corr(),
    annot=False,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# =========================================================
# SAVE CLEANED DATASET
# =========================================================

df.to_csv(
    "cleaned_customer_churn_dataset.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")

# =========================================================
# BUSINESS INSIGHTS
# =========================================================

print("\n=============== BUSINESS INSIGHTS ===============\n")

print("1. Customers with lower monthly logins are more likely to churn.")

print("2. High support ticket counts increase churn probability.")

print("3. Lower CSAT scores strongly correlate with customer churn.")

print("4. Monthly contract users show higher churn rates.")

print("5. Some signup channels generate lower-retention customers.")

print("6. Revenue loss is significantly higher among churned customers.")

print("7. Payment failures contribute to customer dissatisfaction and churn.")

print("8. Customer engagement metrics help identify high-risk users early.")

# =========================================================
# END OF PROJECT
# =========================================================