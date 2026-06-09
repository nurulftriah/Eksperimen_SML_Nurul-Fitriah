import pandas as pd
import numpy as np
import os

def generate_credit_scoring_dataset(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_samples + 1)]
    
    # Age: 18 to 80, with some outliers (negative or extremely high)
    age = np.random.randint(18, 80, size=n_samples).astype(float)
    # Introduce outliers
    age[0] = -5.0
    age[1] = 150.0
    age[2] = 125.0
    
    # Annual income: 20000 to 150000, with some outliers and missing values
    annual_income = np.random.normal(65000, 25000, size=n_samples)
    annual_income = np.clip(annual_income, 15000, 250000).astype(float)
    annual_income[3] = 1200000.0  # extreme outlier
    annual_income[4] = 850000.0   # outlier
    
    # Credit score: 300 to 850
    credit_score = np.random.randint(300, 851, size=n_samples)
    
    # Loan amount requested
    loan_amount = (annual_income * np.random.uniform(0.1, 0.5, size=n_samples)).round(-2)
    # Fix the outlier loan amounts to be reasonable
    loan_amount[3] = 400000.0
    loan_amount[4] = 250000.0
    
    # Marital status: Categorical ('Single', 'Married', 'Divorced')
    marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=n_samples, p=[0.4, 0.45, 0.15])
    
    # Education level: Categorical ('High School', 'Bachelor', 'Master', 'PhD')
    education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])
    
    # Generate target 'default' based on credit score, annual income, age, loan amount
    # Probability of default
    prob = 1 / (1 + np.exp(
        0.01 * (credit_score - 600) + 
        0.00002 * (annual_income - 60000) - 
        0.00005 * loan_amount + 
        0.02 * (age - 35)
    ))
    # Map probability to default (0 or 1)
    default = (np.random.rand(n_samples) > prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'annual_income': annual_income,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'marital_status': marital_status,
        'education_level': education_level,
        'default': default
    })
    
    # Introduce missing values (NaN)
    # age missing
    missing_age_idx = np.random.choice(range(n_samples), size=30, replace=False)
    df.loc[missing_age_idx, 'age'] = np.nan
    
    # annual_income missing
    missing_income_idx = np.random.choice(range(n_samples), size=40, replace=False)
    df.loc[missing_income_idx, 'annual_income'] = np.nan
    
    # marital_status missing
    missing_marital_idx = np.random.choice(range(n_samples), size=25, replace=False)
    df.loc[missing_marital_idx, 'marital_status'] = np.nan
    
    # Introduce duplicate rows (say, 15 duplicate rows)
    dup_indices = np.random.choice(range(n_samples), size=15, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    # Change customer id slightly or keep same to make it true duplicates
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs('namadataset_raw', exist_ok=True)
    
    # Save to CSV
    df.to_csv('namadataset_raw/credit_scoring_raw.csv', index=False)
    print(f"Generated raw credit scoring dataset with {len(df)} rows.")

if __name__ == "__main__":
    generate_credit_scoring_dataset()
