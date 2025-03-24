import pandas as pd
import sqlite3
from datetime import datetime

def parse_datetime(x):
    try:
        # Try the full timestamp format first
        return pd.to_datetime(x, format='%d-%b-%y %I.%M.%S.%f %p')
    except:
        try:
            # Try just the date format
            return pd.to_datetime(x, format='%d-%b-%y')
        except:
            return None

# Read CSV files
print("Loading data...")
loans_df = pd.read_csv('loans.csv')
repayments_df = pd.read_csv('repayments.csv')

# Create SQLite database and connect
print("Creating database...")
conn = sqlite3.connect('loan_analysis.db')

# Convert dates in loans_df
print("Processing loan dates...")
loans_df['disb_date'] = pd.to_datetime(loans_df['disb_date'], format='%d-%b-%y')
loans_df['tenure'] = loans_df['tenure'].str.replace(' days', '').astype(int)

# Convert dates in repayments_df using custom parser
print("Processing repayment dates...")
repayments_df['date_time'] = repayments_df['date_time'].apply(parse_datetime)

# Write to SQLite
print("Saving to database...")
loans_df.to_sql('loans', conn, if_exists='replace', index=False)
repayments_df.to_sql('repayments', conn, if_exists='replace', index=False)

print("Database created successfully!")
conn.close()
