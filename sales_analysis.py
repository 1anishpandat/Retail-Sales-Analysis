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


# ══════════════════════════════════════════════════════════════════
# SECTION 2: AFTERNOON - Data Cleaning & Preparation
# ══════════════════════════════════════════════════════════════════
print("\n" + "━" * 65)
print("  SECTION 2: DATA CLEANING & PREPARATION")
print("━" * 65)

df_clean = df.copy()

# ─── 2.1 Fix date column ─────────────────────────────────────────
print("\n🔧 Step 1: Converting date column...")
df_clean['order_date'] = pd.to_datetime(df_clean['order_date'])

# Extract date parts - very useful for analysis!
df_clean['year']          = df_clean['order_date'].dt.year
df_clean['month']         = df_clean['order_date'].dt.month
df_clean['month_name']    = df_clean['order_date'].dt.strftime('%b')
df_clean['quarter']       = df_clean['order_date'].dt.quarter
df_clean['quarter_label'] = 'Q' + df_clean['quarter'].astype(str)
df_clean['day_of_week']   = df_clean['order_date'].dt.day_name()
df_clean['week_of_year']  = df_clean['order_date'].dt.isocalendar().week.astype(int)
df_clean['year_month']    = df_clean['order_date'].dt.to_period('M').astype(str)
print("   ✓ Date parts extracted: year, month, quarter, day_of_week")

# ─── 2.2 Handle missing values ───────────────────────────────────
print("\n🔧 Step 2: Handling missing values...")

# Missing city → fill with 'Unknown'
missing_city = df_clean['city'].isnull().sum()
df_clean['city']  = df_clean['city'].fillna('Unknown')
df_clean['state'] = df_clean['state'].fillna('Unknown')
print(f"   ✓ City: filled {missing_city} missing values with 'Unknown'")

# Missing discount_pct → fill with 0 (no discount)
missing_disc = df_clean['discount_pct'].isnull().sum()
df_clean['discount_pct'] = df_clean['discount_pct'].fillna(0)
print(f"   ✓ Discount %: filled {missing_disc} missing values with 0 (no discount)")

# Verify no missing values remain
remaining_missing = df_clean.isnull().sum().sum()
print(f"   ✓ Remaining missing values: {remaining_missing}")
# ─── 2.3 Create derived/calculated columns ───────────────────────
print("\n🔧 Step 3: Creating derived columns...")

# Profit margin (assume 35% base cost for Electronics, 60% for others)
cost_pct = df_clean['category'].map({
    'Electronics': 0.65,
    'Furniture':   0.60,
    'Books':       0.50,
    'Accessories': 0.45,
    'Stationery':  0.40
})
df_clean['cost']          = (df_clean['revenue'] * cost_pct).round(2)
df_clean['profit']        = (df_clean['revenue'] - df_clean['cost']).round(2)
df_clean['profit_margin'] = ((df_clean['profit'] / df_clean['revenue']) * 100).round(2)

# Revenue bins for customer value segmentation
df_clean['order_value_tier'] = pd.cut(
    df_clean['revenue'],
    bins   = [0, 1000, 5000, 20000, float('inf')],
    labels = ['Low (<₹1K)', 'Medium (₹1K-5K)', 'High (₹5K-20K)', 'Premium (>₹20K)']
)

# Flag high-discount orders (>20% discount)
df_clean['is_high_discount'] = (df_clean['discount_pct'] > 0.20).astype(int)

print("   ✓ profit, profit_margin columns created")
print("   ✓ order_value_tier column created")
print("   ✓ is_high_discount flag created")

# ─── 2.4 Final clean dataset summary ─────────────────────────────
print(f"\n✅ Data Cleaning Complete!")
print(f"   Rows: {len(df_clean):,} (no rows dropped)")
print(f"   Columns: {len(df_clean.columns)} (was {len(df.columns)}, added {len(df_clean.columns)-len(df.columns)} derived)")
print(f"   Date range: {df_clean['order_date'].min().date()} → {df_clean['order_date'].max().date()}")

# ══════════════════════════════════════════════════════════════════
# SECTION 3: EVENING - KPI Calculation & Basic Analysis
# ══════════════════════════════════════════════════════════════════
print("\n" + "━" * 65)
print("  SECTION 3: KEY PERFORMANCE INDICATORS (KPIs)")
print("━" * 65)

total_revenue   = df_clean['revenue'].sum()
total_orders    = len(df_clean)
total_profit    = df_clean['profit'].sum()
avg_order_value = df_clean['revenue'].mean()
total_units     = df_clean['quantity'].sum()
avg_margin      = df_clean['profit_margin'].mean()
return_rate     = df_clean['is_returned'].mean() * 100
discount_rate   = (df_clean['discount_pct'] > 0).mean() * 100

print(f"""
┌─────────────────────────────────────────────────────────┐
│              OVERALL BUSINESS KPIs (2023-2024)          │
├─────────────────────────────────────────────────────────┤
│  💰 Total Revenue:       ₹{total_revenue:>15,.0f}            │
│  📦 Total Orders:        {total_orders:>15,}            │
│  📈 Total Profit:        ₹{total_profit:>15,.0f}            │
│  🛒 Avg Order Value:     ₹{avg_order_value:>15,.2f}            │
│  📊 Avg Profit Margin:   {avg_margin:>14.1f}%            │
│  📦 Total Units Sold:    {total_units:>15,}            │
│  🔄 Return Rate:         {return_rate:>14.1f}%            │
│  🏷️  Orders with Discount: {discount_rate:>12.1f}%            │
└─────────────────────────────────────────────────────────┘""")