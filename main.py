import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. DATA GENERATION (Synthetic)
# ==========================================
print("Generating Data...")
np.random.seed(42)
weeks = 104
dates = pd.date_range(start='2023-01-01', periods=weeks, freq='W')

df = pd.DataFrame({'Date': dates})
df['TV'] = np.random.randint(1000, 50000, size=weeks)
df['Social'] = np.random.randint(500, 20000, size=weeks)
df['Seasonality'] = 1 + 0.2 * np.sin(2 * np.pi * df.index / 52)

# True hidden logic 
df['Sales'] = (df['TV'] * 0.3) + (df['Social'] * 0.8) + (df['Seasonality'] * 1000) + np.random.normal(0, 500, weeks)

# SCALING
max_sales = df['Sales'].max()
max_tv = df['TV'].max()
max_social = df['Social'].max()

y_obs = df['Sales'].values / max_sales
x_tv = df['TV'].values / max_tv
x_social = df['Social'].values / max_social
x_season = df['Seasonality'].values 

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def geometric_adstock_pymc(x, alpha, l_max=12):
    cycles = np.arange(l_max)
    w = alpha ** cycles 
    w = w / pm.math.sum(w) 
    adstocked = []
    for i in range(len(x)):
        window = x[max(0, i-l_max+1):i+1]
        weights = w[-(len(window)):]
        adstocked.append(pm.math.sum(window * weights))
    return pm.math.stack(adstocked)

def geometric_adstock_numpy(x, alpha, l_max=12):
    cycles = np.arange(l_max)
    w = np.power(alpha, cycles)
    w = w / np.sum(w)
    adstocked = []
    for i in range(len(x)):
        window = x[max(0, i-l_max+1):i+1]
        weights = w[-(len(window)):]
        adstocked.append(np.sum(window * weights))
    return np.array(adstocked)

def saturation_hill_pymc(x, k, s):
    return 1 / (1 + (k / (x + 1e-9))**s)

# ==========================================
# 3. THE BAYESIAN MODEL
# ==========================================
print("Building & Training Model (Wait ~30s)...")

with pm.Model() as mmm_model:
    intercept = pm.Normal("Intercept", mu=0, sigma=1)
    beta_season = pm.Normal("beta_season", mu=0, sigma=1)
    beta_tv = pm.HalfNormal("beta_tv", sigma=1)
    beta_social = pm.HalfNormal("beta_social", sigma=1)
    alpha_tv = pm.Beta("alpha_tv", alpha=3, beta=3)
    k_tv = pm.Beta("k_tv", alpha=2, beta=2)
    k_social = pm.Beta("k_social", alpha=2, beta=2)
    
    tv_adstock = geometric_adstock_pymc(x_tv, alpha_tv)
    tv_saturated = saturation_hill_pymc(tv_adstock, k_tv, s=1)
    social_saturated = saturation_hill_pymc(x_social, k_social, s=1)
    
    mu = (intercept + beta_season * x_season + beta_tv * tv_saturated + beta_social * social_saturated)
    sigma = pm.HalfNormal("sigma", sigma=1)
    likelihood = pm.StudentT("obs", nu=4, mu=mu, sigma=sigma, observed=y_obs)
    
    trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=True)

# ==========================================
# 4. DATA EXTRACTION FOR REPORTING
# ==========================================
post = trace.posterior.mean(dim=["chain", "draw"])
b_tv, b_soc = float(post["beta_tv"]), float(post["beta_social"])
k_tv_val, k_soc_val = float(post["k_tv"]), float(post["k_social"])
alpha_tv_val = float(post["alpha_tv"])
intercept_val = float(post["Intercept"])
season_val = float(post["beta_season"])

# --- CALCULATE TIME-SERIES ARRAYS (REQUIRED FOR CHART) ---
# We need these arrays to plot the Waterfall chart
tv_input_adstocked = geometric_adstock_numpy(x_tv, alpha_tv_val)
tv_contrib_ts = b_tv * (1 / (1 + (k_tv_val / tv_input_adstocked)**1)) * max_sales
soc_contrib_ts = b_soc * (1 / (1 + (k_soc_val / x_social)**1)) * max_sales
base_contrib_ts = (intercept_val + season_val * x_season) * max_sales

# Calculate Totals for Strategy
tv_rev = np.sum(tv_contrib_ts)
soc_rev = np.sum(soc_contrib_ts)
total_tv_spend = np.sum(df['TV'])
total_soc_spend = np.sum(df['Social'])

# ROI Calculation (Return on Investment)
tv_roi = tv_rev / total_tv_spend
soc_roi = soc_rev / total_soc_spend

# ==========================================
# 5. AUTOMATED STRATEGY ENGINE
# ==========================================
print("\n" + "="*40)
print("     AI STRATEGIC RECOMMENDATION")
print("="*40)

# A. Determine Winner and Loser
if soc_roi > tv_roi:
    winner, loser = "Social", "TV"
    winner_roi, loser_roi = soc_roi, tv_roi
else:
    winner, loser = "TV", "Social"
    winner_roi, loser_roi = tv_roi, soc_roi

# B. Simulate a 5% Budget Shift
shift_amount = (total_tv_spend + total_soc_spend) * 0.05 
predicted_loss = shift_amount * loser_roi
predicted_gain = shift_amount * winner_roi
net_profit = predicted_gain - predicted_loss

print(f"1. PERFORMANCE AUDIT:")
print(f"   - TV ROI:     {tv_roi:.2f}x (For every $1 spent, you get ${tv_roi:.2f})")
print(f"   - Social ROI: {soc_roi:.2f}x (For every $1 spent, you get ${soc_roi:.2f})")
print(f"   - VERDICT:    {winner} is performing {((winner_roi/loser_roi)-1)*100:.0f}% better than {loser}.")

print(f"\n2. STRATEGY PROPOSAL:")
print(f"   \"Shift budget from {loser} to {winner}.\"")
print(f"   We recommend moving 5% of your total budget (${shift_amount:,.0f}) from {loser} to {winner}.")

print(f"\n3. PREDICTED BUSINESS IMPACT:")
print(f"   If you implement this change next quarter:")
print(f"   - Revenue lost from {loser}: -${predicted_loss:,.0f}")
print(f"   - Revenue gained from {winner}: +${predicted_gain:,.0f}")
print(f"   -------------------------------------------")
print(f"   - NET PROFIT INCREASE:      +${net_profit:,.0f}")
print("="*40 + "\n")

# ==========================================
# 6. VISUALIZATION (Structured)
# ==========================================
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})

# CHANGED: 3 Rows instead of 2 to fit the Waterfall Chart
fig = plt.figure(figsize=(15, 14))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.8])

# --- ROW 1: SALES DECOMPOSITION (Added as requested) ---
ax1 = fig.add_subplot(gs[0, :])
dates_arr = df['Date']
ax1.stackplot(dates_arr, 
              base_contrib_ts, 
              tv_contrib_ts, 
              soc_contrib_ts, 
              labels=['Base/Organic', 'TV Lift', 'Social Lift'],
              colors=['#e0e0e0', '#1f77b4', '#ff7f0e'], 
              alpha=0.8)
ax1.set_title("1. Sales Drivers: How much revenue did Marketing actually add?", fontsize=14, fontweight='bold')
ax1.set_ylabel("Revenue ($)")
ax1.legend(loc='upper left')

# --- ROW 2: DIAGNOSTICS (ROI & Saturation) ---
# Chart 2: ROI Comparison
ax2 = fig.add_subplot(gs[1, 0])
bars = ax2.bar(['TV', 'Social'], [tv_roi, soc_roi], color=['#1f77b4', '#ff7f0e'])
ax2.bar_label(bars, fmt='%.2fx')
ax2.set_title("2a. Efficiency Check: ROI per Channel", fontweight='bold')
ax2.set_ylabel("Return on Ad Spend (ROAS)")

# Chart 3: Saturation Curves
ax3 = fig.add_subplot(gs[1, 1])
spend_sim = np.linspace(0, 1.5, 100) 
tv_curve = b_tv * (1 / (1 + (k_tv_val / spend_sim)**1)) * max_sales
soc_curve = b_soc * (1 / (1 + (k_soc_val / spend_sim)**1)) * max_sales
ax3.plot(spend_sim*100, tv_curve, label='TV', color='#1f77b4', linewidth=3)
ax3.plot(spend_sim*100, soc_curve, label='Social', color='#ff7f0e', linewidth=3)
ax3.set_title("2b. Saturation: Where is the limit?", fontweight='bold')
ax3.set_xlabel("Spend Intensity (%)")
ax3.legend()

# --- ROW 3: STRATEGY (Optimization) ---
ax4 = fig.add_subplot(gs[2, :])
scenarios = ['Current Strategy', 'Optimized Strategy (+5% Shift)']
values = [tv_rev + soc_rev, (tv_rev + soc_rev) + net_profit]
colors = ['gray', 'green']
bars2 = ax4.barh(scenarios, values, color=colors)
ax4.bar_label(bars2, fmt='$%.0f', padding=5)
ax4.set_title(f"3. Forecast: Impact of Moving Budget to {winner}", fontweight='bold', color='green')
ax4.set_xlabel("Total Revenue ($)")
ax4.set_xlim(min(values)*0.95, max(values)*1.05)

plt.tight_layout()
plt.show()