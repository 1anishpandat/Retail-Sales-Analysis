"""
Run this FIRST to generate your dataset.
Command: python generate_data.py
"""
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

products = [
    ("P001","Laptop Pro 15","Electronics",75000,0.15),
    ("P002","Wireless Mouse","Electronics",1500,0.20),
    ("P003","USB-C Hub","Electronics",2000,0.18),
    ("P004","Mechanical Keyboard","Electronics",4500,0.22),
    ("P005","Monitor 24 inch","Electronics",18000,0.12),
    ("P006","Webcam HD 1080p","Electronics",3500,0.19),
    ("P007","Headphones Pro","Electronics",5500,0.25),
    ("P008","Phone Stand","Accessories",599,0.30),
    ("P009","Cable Organizer","Accessories",299,0.35),
    ("P010","Laptop Bag","Accessories",1800,0.28),
    ("P011","Screen Cleaner","Accessories",199,0.40),
    ("P012","Python Book","Books",599,0.10),
    ("P013","SQL for Analysts","Books",499,0.12),
    ("P014","Data Science Guide","Books",799,0.08),
    ("P015","Excel Mastery","Books",449,0.15),
    ("P016","Office Chair Ergonomic","Furniture",12000,0.10),
    ("P017","Standing Desk","Furniture",25000,0.08),
    ("P018","Monitor Stand","Furniture",2500,0.20),
    ("P019","Desk Lamp LED","Furniture",1800,0.22),
    ("P020","Notebook A4 Pack","Stationery",299,0.50),
]

cities = [
    ("Mumbai","Maharashtra",0.22),
    ("Delhi","Delhi",0.18),
    ("Bangalore","Karnataka",0.16),
    ("Chennai","Tamil Nadu",0.12),
    ("Pune","Maharashtra",0.10),
    ("Hyderabad","Telangana",0.09),
    ("Kolkata","West Bengal",0.08),
    ("Ahmedabad","Gujarat",0.05),
]

payment_methods = ["Credit Card","Debit Card","UPI","Net Banking","Cash on Delivery"]
channels        = ["Online","In-Store","Mobile App"]
segments        = ["Consumer","Corporate","Home Office"]

rows = []
order_id   = 1000
start_date = datetime(2023, 1, 1)
end_date   = datetime(2024, 12, 31)
date_range = (end_date - start_date).days

for _ in range(5500):
    city_data  = random.choices(cities,    weights=[c[2]  for c in cities])[0]
    prod_w     = [3,3,2,2,2,2,2,2,1,2,1,1,1,1,1,1,1,1,1,2]
    prod_data  = random.choices(products,  weights=prod_w)[0]

    order_date = start_date + timedelta(days=random.randint(0, date_range))
    month      = order_date.month
    qty_max    = 5 if month in [10,11,12] else (3 if month in [6,7,8] else 4)
    quantity   = random.randint(1, qty_max)
    base_price = prod_data[4]

    disc_pct   = round(random.uniform(0, prod_data[4]) * random.choice([0,0,0,1]), 2)
    disc_amt   = round(base_price * disc_pct, 2)
    sale_price = round(base_price - disc_amt, 2)
    revenue    = round(sale_price * quantity, 2)

    if random.random() < 0.03: disc_pct = None
    city_val   = city_data[0] if random.random() > 0.02 else None

    rows.append({
        "order_id":          f"ORD-{order_id}",
        "order_date":        order_date.strftime("%Y-%m-%d"),
        "product_id":        prod_data[0],
        "product_name":      prod_data[1],
        "category":          prod_data[2],
        "city":              city_val,
        "state":             city_data[1] if city_val else None,
        "customer_segment":  random.choices(segments,        weights=[50,30,20])[0],
        "quantity":          quantity,
        "unit_price":        base_price,
        "discount_pct":      disc_pct,
        "discount_amt":      disc_amt,
        "sale_price":        sale_price,
        "revenue":           revenue,
        "payment_method":    random.choices(payment_methods, weights=[25,20,30,15,10])[0],
        "channel":           random.choices(channels,        weights=[50,30,20])[0],
        "is_returned":       1 if random.random() < 0.05 else 0,
    })
    order_id += 1

df = pd.DataFrame(rows)
os.makedirs('data', exist_ok=True)
df.to_csv('data/retail_sales_raw.csv', index=False)

print("=" * 50)
print("  DATA GENERATED SUCCESSFULLY!")
print("=" * 50)
print(f"  Rows:       {len(df):,}")
print(f"  Columns:    {len(df.columns)}")
print(f"  Date Range: {df['order_date'].min()} → {df['order_date'].max()}")
print(f"  Saved to:   data/retail_sales_raw.csv")
print("=" * 50)
print("  Now run: python sales_analysis.py")
print("=" * 50)
