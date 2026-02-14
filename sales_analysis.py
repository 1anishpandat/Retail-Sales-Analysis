"""
╔══════════════════════════════════════════════════════════════════╗
║        PROJECT 1: RETAIL SALES PERFORMANCE ANALYSIS             ║
║        Python | Pandas | NumPy | Matplotlib | Seaborn           ║
║        Analysis Period: January 2023 - December 2024            ║
╚══════════════════════════════════════════════════════════════════╝

BUSINESS QUESTIONS WE WILL ANSWER:
Q1. What is the overall revenue performance and growth trend?
Q2. Which product categories drive the most revenue?
Q3. Which cities and regions are top performers?
Q4. What are the monthly and seasonal sales patterns?
Q5. Which products are the best and worst performers?
Q6. How do different sales channels compare?
Q7. What is the impact of discounts on revenue?
"""

# ══════════════════════════════════════════════════════════════════
# SECTION 0: SETUP - Import libraries
# ══════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Style settings - makes charts look professional
plt.rcParams['figure.figsize']  = (12, 6)
plt.rcParams['font.family']     = 'DejaVu Sans'
plt.rcParams['axes.spines.top']    = False
plt.rcParams['axes.spines.right']  = False
sns.set_palette("husl")

print("=" * 65)
print("  PROJECT 1: RETAIL SALES PERFORMANCE ANALYSIS")
print("=" * 65)
print("✓ Libraries loaded successfully\n")


# ══════════════════════════════════════════════════════════════════
# SECTION 1: MORNING - Load & Explore Data
# ══════════════════════════════════════════════════════════════════
print("━" * 65)
print("  SECTION 1: DATA LOADING & EXPLORATION")
print("━" * 65)

# ─── 1.1 Load the dataset ────────────────────────────────────────
df = pd.read_csv('data/retail_sales_raw.csv')

print(f"\n📂 Dataset loaded!")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─── 1.2 First look at data ──────────────────────────────────────
print("\n📋 First 5 rows:")
print(df.head().to_string())

# ─── 1.3 Data types and structure ────────────────────────────────
print("\n🔍 Column Info:")
print(f"{'Column':<20} {'Dtype':<15} {'Non-Null Count':<15} {'Sample'}")
print("-" * 70)
for col in df.columns:
    dtype    = str(df[col].dtype)
    non_null = df[col].count()
    sample   = str(df[col].dropna().iloc[0]) if non_null > 0 else "N/A"
    print(f"{col:<20} {dtype:<15} {non_null:<15,} {sample[:30]}")

# ─── 1.4 Statistical summary ─────────────────────────────────────
print("\n📊 Statistical Summary (Numeric Columns):")
print(df.describe().round(2).to_string())

# ─── 1.5 Missing values check ────────────────────────────────────
print("\n🔎 Missing Values Check:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("   No missing values found!")

# ─── 1.6 Duplicates check ────────────────────────────────────────
dupes = df.duplicated().sum()
print(f"\n🔎 Duplicate Rows: {dupes:,}")

# ─── 1.7 Unique value counts ─────────────────────────────────────
print("\n🔎 Unique Values per Category Column:")
cat_cols = ['category', 'city', 'customer_segment', 'payment_method', 'channel']
for col in cat_cols:
    uniq = df[col].nunique()
    vals = df[col].dropna().unique()[:5]
    print(f"   {col}: {uniq} unique → {list(vals)}")

