# Bayesian Market Mix Modeling (MMM) & Optimization Engine

### Project Overview
This project implements a **Bayesian Structural Time-Series Model** to solve the "Marketing Attribution" problem. Using **PyMC** and **Probabilistic Programming**, it quantifies the ROI of media channels (TV vs. Social) while accounting for real-world complexities like **Adstock (Memory Effect)** and **Diminishing Returns (Saturation)**.

Unlike standard OLS regression, this model provides a "Strategy Engine" that automatically recommends budget shifts to maximize Net Profit.

### Tech Stack
* **Modeling:** Python, PyMC (Bayesian Inference), MCMC Sampling (NUTS)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Concepts:** Geometric Adstock, Hill Function Saturation, ROI Analysis

### Key Features
1.  **Adstock Modeling:** Learns the decay rate ($\alpha$) of ads (e.g., TV lasts weeks, Social lasts days).
2.  **Saturation Curves:** Uses Hill Functions to detect when spending becomes wasteful.
3.  **Automated Strategy:** Mathematically calculates the optimal budget split.
4.  **Executive Dashboard:** Generates waterfall charts and saturation curves for stakeholders.

### Results (Simulation)
* **Insight:** The model identified that TV spend was saturated (flat curve), while Social Media had high uncaptured efficiency.
* **Action:** Recommended a **5% Budget Shift** from TV to Social.
* **Impact:** Predicted a **Net Profit increase** of ~10% without increasing total spend.

### How to Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the script:
   `python main.py`