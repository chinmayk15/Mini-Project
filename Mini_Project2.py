import pandas as pd
import numpy as np
import random
from datetime import datetime

random.seed(42)
np.random.seed(42)


def load_data(filepath: str) -> pd.DataFrame:
  


    df = pd.read_csv("C:/Users/Chinmay/OneDrive/Documents/E-commerece sales data 2024.csv")

   
    df = df.dropna(axis=1, how="all")

  
    df.columns = ["user_id", "product_id", "interaction_type", "timestamp"]


    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")

  
    df = df.dropna(subset=["product_id", "interaction_type"])

    print(f" Loaded {len(df):,} interactions | "
          f"{df['product_id'].nunique():,} unique products | "
          f"{df['user_id'].nunique():.0f} unique users\n")

    return df



INTERACTION_WEIGHTS = {
    "purchase": 3.0,
    "like":     1.5,
    "view":     1.0,
}

def calculate_demand_scores(df: pd.DataFrame) -> pd.DataFrame:

    print(" Calculating demand scores...")


    df["weight"] = df["interaction_type"].map(INTERACTION_WEIGHTS).fillna(0)


    demand = (
        df.groupby("product_id").agg(total_interactions=("interaction_type", "count"),demand_score=("weight", "sum"),purchase_count=("interaction_type", lambda x: (x == "purchase").sum()),view_count=("interaction_type", lambda x: (x == "view").sum()),like_count=("interaction_type", lambda x: (x == "like").sum()),).reset_index()
    )
    min_score = demand["demand_score"].min()
    max_score = demand["demand_score"].max()

    if max_score > min_score:
        demand["demand_index"] = ((demand["demand_score"] - min_score) / (max_score - min_score) * 100).round(1)
    else:
        demand["demand_index"] = 50.0   

    print(f"   Top demand score : {demand['demand_score'].max():.1f}")
    print(f"   Avg demand score : {demand['demand_score'].mean():.1f}\n")

    return demand


def simulate_market_data(demand_df: pd.DataFrame) -> pd.DataFrame:
    
    print(" Simulating market data (prices, inventory, competitors)...")

    n = len(demand_df)

  
    demand_df["base_price"] = np.random.randint(199, 9999, size=n)


    demand_df["inventory_pct"] = np.random.randint(5, 95, size=n)

    competitor_shift = np.random.uniform(-0.15, 0.15, size=n)
    demand_df["competitor_price"] = (
        demand_df["base_price"] * (1 + competitor_shift)
    ).round(0).astype(int)

    print(f"   Price range : ₹{demand_df['base_price'].min():,} – "
          f"₹{demand_df['base_price'].max():,}")
    print(f"   Avg inventory : {demand_df['inventory_pct'].mean():.1f}%\n")

    return demand_df


BASE_ELASTICITY = -1.5       

def compute_price_adjustment(row: pd.Series) -> dict:

    adjustment=0
    demand_index  = row["demand_index"]       
    inventory_pct = row["inventory_pct"]       
    base_price    = row["base_price"]          
    comp_price    = row["competitor_price"]    

    if demand_index > 70:
        elasticity = BASE_ELASTICITY * 0.6    
        demand_adj = +0.12                    
    elif demand_index < 30:
        elasticity = BASE_ELASTICITY * 1.4   
        demand_adj = -0.10                   
    else:
        elasticity = BASE_ELASTICITY
        demand_adj = +0.02                   
   
    if inventory_pct < 15:
        adjustment += 0.08
        signal = "Flash/Low-stock"
    elif inventory_pct > 80:
        adjustment -= 0.12
        signal = "Overstock clearance"
    elif demand_index > 70:
        signal = "High demand"
    elif demand_index < 30:
        signal = "Low demand"
    else:
        signal = "Optimal hold"
    
    if comp_price > 0:
        comp_gap_pct = (base_price - comp_price) / comp_price  
        if comp_gap_pct > 0.10:
            adjustment = adjustment - 0.05
        elif comp_gap_pct < -0.10:
            adjustment = adjustment + 0.04
    
    adjustment = max(-0.25, min(0.25, adjustment))
    recommended_price = round(base_price * (1 + adjustment))
    return {
        "price_change_pct": round(adjustment * 100, 1),
        "recommended_price": recommended_price,
        "signal": signal,
        "elasticity_used": round(elasticity, 2),
    }


def apply_pricing_engine(df: pd.DataFrame) -> pd.DataFrame:
    
    print("Running pricing engine on all products...")

    results = df.apply(compute_price_adjustment, axis=1, result_type="expand")
    df = pd.concat([df, results], axis=1)

    df["estimated_revenue_delta_pct"] = (
        df["price_change_pct"] * (1 + 1 / abs(df["elasticity_used"]))
    ).round(1)

    total_repriced = (df["price_change_pct"] != 0).sum()
    print(f"   Products repriced : {total_repriced:,} / {len(df):,}")
    print(f"   Avg price change  : {df['price_change_pct'].mean():+.1f}%\n")

    return df


FLASH_SALE_DEMAND_THRESHOLD   = 75  
FLASH_SALE_INVENTORY_THRESHOLD = 20 

def detect_flash_sales(df: pd.DataFrame) -> pd.DataFrame:
    
    df["is_flash_sale"] = (
        (df["demand_index"] >= FLASH_SALE_DEMAND_THRESHOLD) &
        (df["inventory_pct"] <= FLASH_SALE_INVENTORY_THRESHOLD)
    )

    flash_count = df["is_flash_sale"].sum()
    print(f"Flash sales detected: {flash_count} products\n")

    return df


def print_report(df: pd.DataFrame, top_n: int = 15):
    
    print("=" * 70)
    print("    DYNAMIC PRICING ENGINE — LIVE REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    avg_uplift     = df["price_change_pct"].mean()
    stockout_risk  = (df["inventory_pct"] < 15).mean() * 100
    overstock_risk = (df["inventory_pct"] > 80).mean() * 100
    flash_count    = df["is_flash_sale"].sum()

    print("\n  KEY PERFORMANCE INDICATORS")
    print(f"  {'Avg price adjustment':<30}: {avg_uplift:+.1f}%")
    print(f"  {'Products at stockout risk (<15%)':<30}: {stockout_risk:.1f}% of catalog")
    print(f"  {'Products overstocked (>80%)':<30}: {overstock_risk:.1f}% of catalog")
    print(f"  {'Flash sale alerts':<30}: {flash_count} products")
    print(f"  {'Estimated avg revenue uplift':<30}: {df['estimated_revenue_delta_pct'].mean():+.1f}%")

    flash_df = df[df["is_flash_sale"]].sort_values("demand_index", ascending=False)
    if not flash_df.empty:
        print("\n FLASH SALE ALERTS")
        print(f"  {'Product ID':<36} {'Demand':>7} {'Inv%':>6} {'Old Price':>10} {'New Price':>10}")
        print("  " + "-" * 72)
        for _, row in flash_df.head(5).iterrows():
            print(f"  {row['product_id']:<36} "
                  f"{row['demand_index']:>6.1f} "
                  f"{row['inventory_pct']:>5}% "
                  f"₹{row['base_price']:>9,} "
                  f"₹{row['recommended_price']:>9,}")
    print(f"\n  TOP {top_n} PRODUCTS BY DEMAND INDEX")
    print(f"  {'Product ID':<36} {'Dmnd':>5} {'Inv%':>5} "f"{'Old ₹':>8} {'New ₹':>8} {'Chg%':>6} {'Signal':<22}")
    print("  " + "-" * 96)

    top = df.sort_values("demand_index", ascending=False).head(top_n)
    for _, row in top.iterrows():
        arrow = "↑" if row["price_change_pct"] > 0 else ("↓" if row["price_change_pct"] < 0 else "→")
        print(f"  {row['product_id']:<36} "
              f"{row['demand_index']:>5.1f} "
              f"{row['inventory_pct']:>4}% "
              f"₹{row['base_price']:>7,} "
              f"₹{row['recommended_price']:>7,} "
              f"{arrow}{abs(row['price_change_pct']):>4.1f}% "
              f"  {row['signal']}")
    print("\n  PRICING SIGNAL BREAKDOWN")
    signal_counts = df["signal"].value_counts()
    for signal, count in signal_counts.items():
        bar = " " * int(count / len(df) * 40)
        print(f"  {signal:<25} {count:>5} products  {bar}")

    print("\n" + "=" * 70)
    print(" Engine run complete. Prices ready to push to Redis cache.")
    print("=" * 70 + "\n")


def save_results(df: pd.DataFrame, output_path: str):
    
    output_cols = [
        "product_id",
        "demand_index",
        "purchase_count",
        "view_count",
        "like_count",
        "inventory_pct",
        "base_price",
        "competitor_price",
        "recommended_price",
        "price_change_pct",
        "estimated_revenue_delta_pct",
        "signal",
        "is_flash_sale",
        "elasticity_used",
    ]
    df[output_cols].to_csv(output_path, index=False)
    print(f" Results saved to: {output_path}\n")

def main():
    INPUT_CSV  = "E-commerece_sales_data_2024.csv"  
    OUTPUT_CSV = "pricing_decisions.csv"              

    print("\n" + "=" * 70)
    print("  AI Dynamic Pricing Engine — Starting up")
    print("=" * 70 + "\n")

    df_raw = load_data(INPUT_CSV)

    df_demand = calculate_demand_scores(df_raw)

    df_market = simulate_market_data(df_demand)

    df_priced = apply_pricing_engine(df_market)

    df_final = detect_flash_sales(df_priced)

    print_report(df_final, top_n=15)

    save_results(df_final, OUTPUT_CSV)

    output=pd.read_csv(OUTPUT_CSV)
   
    print(output)


if __name__ == "__main__":
    main()
