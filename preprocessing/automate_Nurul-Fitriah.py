import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the raw dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def clean_data(df):
    """Perform data cleaning: duplicates, missing values, and outliers."""
    df_cleaned = df.copy()
    
    # 1. Remove duplicate rows
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    removed_dups = initial_rows - len(df_cleaned)
    print(f"Removed {removed_dups} duplicate rows.")
    
    # 2. Handle missing values
    # Age: Fill missing with median
    age_median = df_cleaned['age'].median()
    df_cleaned['age'] = df_cleaned['age'].fillna(age_median)
    
    # Annual Income: Fill missing with median
    income_median = df_cleaned['annual_income'].median()
    df_cleaned['annual_income'] = df_cleaned['annual_income'].fillna(income_median)
    
    # Marital Status: Fill missing with mode (most frequent value)
    marital_mode = df_cleaned['marital_status'].mode()[0]
    df_cleaned['marital_status'] = df_cleaned['marital_status'].fillna(marital_mode)
    
    print("Handled missing values (imputed numerical with median, categorical with mode).")
    
    # 3. Handle outliers (IQR Method for age and annual_income)
    # Clip age to realistic range (18 to 100) or IQR
    # Age outlier handling:
    q1_age = df_cleaned['age'].quantile(0.25)
    q3_age = df_cleaned['age'].quantile(0.75)
    iqr_age = q3_age - q1_age
    lower_age = q1_age - 1.5 * iqr_age
    upper_age = q3_age + 1.5 * iqr_age
    # Ensure age is at least 18 and at most 100
    lower_bound_age = max(18, lower_age)
    upper_bound_age = min(100, upper_age)
    df_cleaned['age'] = df_cleaned['age'].clip(lower_bound_age, upper_bound_age)
    
    # Annual income outlier handling:
    q1_inc = df_cleaned['annual_income'].quantile(0.25)
    q3_inc = df_cleaned['annual_income'].quantile(0.75)
    iqr_inc = q3_inc - q1_inc
    lower_inc = q1_inc - 1.5 * iqr_inc
    upper_inc = q3_inc + 1.5 * iqr_inc
    df_cleaned['annual_income'] = df_cleaned['annual_income'].clip(lower_inc, upper_inc)
    
    print("Handled outliers using IQR clipping.")
    return df_cleaned

def preprocess_features(df):
    """Feature engineering, encoding, and scaling."""
    df_preprocessed = df.copy()
    
    # 1. Binning (Age categories)
    # Define bins: 18-35 (Young), 36-55 (Middle-Aged), 56+ (Senior)
    bins = [17, 35, 55, np.inf]
    labels = ['Young', 'Middle-Aged', 'Senior']
    df_preprocessed['age_group'] = pd.cut(df_preprocessed['age'], bins=bins, labels=labels)
    print("Added 'age_group' column using binning.")
    
    # Drop customer_id as it's just an identifier
    if 'customer_id' in df_preprocessed.columns:
        df_preprocessed = df_preprocessed.drop(columns=['customer_id'])
        
    # 2. Categorical Encoding (One-hot encoding)
    categorical_cols = ['marital_status', 'education_level', 'age_group']
    # Perform one-hot encoding
    df_preprocessed = pd.get_dummies(df_preprocessed, columns=categorical_cols, drop_first=True)
    
    # Convert bool columns from get_dummies to int (0 or 1)
    bool_cols = df_preprocessed.select_dtypes(include=['bool']).columns
    df_preprocessed[bool_cols] = df_preprocessed[bool_cols].astype(int)
    print("Encoded categorical variables (one-hot encoding).")
    
    # 3. Normalization/Standardization (Scaling numerical features)
    numerical_cols = ['age', 'annual_income', 'credit_score', 'loan_amount']
    scaler = StandardScaler()
    df_preprocessed[numerical_cols] = scaler.fit_transform(df_preprocessed[numerical_cols])
    print("Standardized numerical features using StandardScaler.")
    
    return df_preprocessed

def save_data(df, output_path):
    """Save the preprocessed dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed dataset to: {output_path} (Shape: {df.shape})")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.abspath(os.path.join(script_dir, "../namadataset_raw/credit_scoring_raw.csv"))
    default_output = os.path.abspath(os.path.join(script_dir, "namadataset_preprocessing/credit_scoring_preprocessing.csv"))
    
    parser = argparse.ArgumentParser(description="Automate Data Preprocessing for Credit Scoring")
    parser.add_argument("--input", default=default_input, help="Path to raw CSV dataset")
    parser.add_argument("--output", default=default_output, help="Path to save preprocessed CSV")
    
    args = parser.parse_args()
    
    # Execute workflow
    df_raw = load_data(args.input)
    df_cleaned = clean_data(df_raw)
    df_preprocessed = preprocess_features(df_cleaned)
    save_data(df_preprocessed, args.output)
    print("Preprocessing workflow completed successfully!")

if __name__ == "__main__":
    main()
