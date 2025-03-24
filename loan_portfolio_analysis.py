import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Read the datasets
print("Loading datasets...")
loans_df = pd.read_csv('loans.csv')
repayments_df = pd.read_csv('repayments.csv')

# Data preprocessing
print("Processing data...")
loans_df['disb_date'] = pd.to_datetime(loans_df['disb_date'], format='%d-%b-%y')
loans_df['tenure'] = loans_df['tenure'].str.replace(' days', '').astype(int)
loans_df['due_date'] = loans_df['disb_date'] + pd.to_timedelta(loans_df['tenure'], unit='D')
loans_df['effective_rate'] = (loans_df['loan_fee'] / loans_df['loan_amount'] * 100).round(2)

# Process repayments - handle mixed date formats
def parse_date(date_str):
    try:
        # Try full datetime format first
        if '.000000000' in date_str:
            date_str = date_str.replace('.000000000', '')
            date_str = date_str.replace('.', ':')
        return pd.to_datetime(date_str, format='%d-%b-%y %I:%M:%S %p')
    except:
        try:
            # Try date-only format
            return pd.to_datetime(date_str, format='%d-%b-%y')
        except:
            # Return NaT for unparseable dates
            return pd.NaT

print("Processing repayment dates...")
repayments_df['date_time'] = repayments_df['date_time'].apply(parse_date)
repayments_df['rep_month'] = pd.to_datetime(repayments_df['rep_month'].astype(str), format='%Y%m')

# 1. Loan Product Analysis
print("\n1. LOAN PRODUCT ANALYSIS")
print("=" * 50)

tenure_stats = loans_df.groupby('tenure').agg({
    'loan_amount': ['count', 'sum', 'mean', 'min', 'max'],
    'loan_fee': ['sum', 'mean'],
    'effective_rate': 'mean'
}).round(2)

print("\nTenure Distribution and Metrics:")
print(tenure_stats)

# Amount brackets analysis
amount_bins = [0, 500, 1000, 2000, 3000, 3500]
loans_df['amount_bracket'] = pd.cut(loans_df['loan_amount'], bins=amount_bins)
amount_dist = loans_df.groupby(['tenure', 'amount_bracket']).size().unstack(fill_value=0)
print("\nLoan Amount Distribution by Tenure:")
print(amount_dist)

# 2. Repayment Analysis
print("\n2. REPAYMENT ANALYSIS")
print("=" * 50)

# Merge loans and repayments
merged_df = pd.merge(loans_df, repayments_df, on='customer_id', how='left')
merged_df['days_to_repay'] = (merged_df['date_time'] - merged_df['disb_date']).dt.days

# Repayment timing analysis
repayment_timing = merged_df.groupby('tenure').agg({
    'days_to_repay': ['mean', 'std', 'min', 'max']
}).round(2)

print("\nRepayment Timing Analysis:")
print(repayment_timing)

# Repayment type distribution
repayment_types = repayments_df['repayment_type'].value_counts()
print("\nRepayment Type Distribution:")
print(repayment_types)

# 3. Performance Analysis
print("\n3. PERFORMANCE ANALYSIS")
print("=" * 50)

# Monthly metrics
loans_df['month'] = loans_df['disb_date'].dt.to_period('M')
monthly_metrics = loans_df.groupby('month').agg({
    'loan_amount': ['count', 'sum'],
    'loan_fee': 'sum'
}).round(2)

print("\nMonthly Performance Metrics:")
print(monthly_metrics)

# Customer behavior
customer_loans = loans_df['customer_id'].value_counts()
print("\nCustomer Behavior:")
print(f"Total unique customers: {loans_df['customer_id'].nunique():,}")
print(f"Customers with multiple loans: {sum(customer_loans > 1):,}")
print(f"Maximum loans per customer: {customer_loans.max():,}")

# 4. Risk Analysis
print("\n4. RISK ANALYSIS")
print("=" * 50)

# Current exposure
reference_date = datetime(2024, 4, 30)
active_loans = loans_df[loans_df['due_date'] > reference_date]

exposure_by_tenure = active_loans.groupby('tenure').agg({
    'loan_amount': ['count', 'sum'],
    'loan_fee': 'sum'
}).round(2)

print(f"\nCredit Exposure as of April 30, 2024:")
print(exposure_by_tenure)

# Repayment performance by loan size
merged_df['loan_size_category'] = pd.cut(merged_df['loan_amount'], 
                                       bins=[0, 500, 1000, 2000, 3500], 
                                       labels=['Small', 'Medium', 'Large', 'Extra Large'])
repayment_by_size = merged_df.groupby('loan_size_category').agg({
    'days_to_repay': ['mean', 'count'],
    'amount': 'sum'
}).round(2)

print("\nRepayment Performance by Loan Size:")
print(repayment_by_size)

# Detailed repayment timing analysis
print("\n=== DETAILED REPAYMENT TIMING ANALYSIS ===")
print("=" * 50)

# Calculate days relative to due date (negative means early payment)
merged_df['days_to_due'] = (merged_df['date_time'] - merged_df['due_date']).dt.days
merged_df['payment_category'] = pd.cut(merged_df['days_to_due'], 
                                     bins=[-float('inf'), -7, -1, 0, 1, 7, float('inf')],
                                     labels=['Very Early (>7 days)', 'Early (1-7 days)', 'On Time', 
                                            'Late (1 day)', 'Late (2-7 days)', 'Very Late (>7 days)'])

# Analyze early payments
early_payments = merged_df[merged_df['days_to_due'] < 0]
print("\nEarly Payment Statistics:")
print("-" * 30)
print(f"Total early payments: {len(early_payments):,} ({len(early_payments)/len(merged_df)*100:.1f}% of all loans)")
print("\nEarly Payment Distribution by Tenure:")
tenure_early = early_payments.groupby('tenure').agg({
    'days_to_due': ['count', 'mean', 'min'],
    'loan_amount': ['mean', 'sum']
}).round(2)
print(tenure_early)

print("\nPayment Timing Distribution:")
timing_dist = merged_df['payment_category'].value_counts().sort_index()
print(timing_dist)

# Analyze payment patterns by loan characteristics
print("\nEarly Payment Analysis by Loan Size and Tenure:")
early_patterns = early_payments.groupby(['tenure', 'loan_size_category']).agg({
    'days_to_due': ['count', 'mean', 'min'],
    'loan_amount': 'mean',
    'repayment_type': lambda x: (x == 'Automatic').mean() * 100  # Percentage of automatic payments
}).round(2)
print(early_patterns)

# Update the report file with early payment analysis
with open('loan_portfolio_report.txt', 'a') as f:
    f.write("\n\nEARLY PAYMENT ANALYSIS\n")
    f.write("=" * 50 + "\n")
    f.write(f"\nTotal early payments: {len(early_payments):,} ({len(early_payments)/len(merged_df)*100:.1f}% of all loans)\n")
    f.write("\nPayment Timing Distribution:\n")
    for category, count in timing_dist.items():
        f.write(f"- {category}: {count:,} loans ({count/len(merged_df)*100:.1f}%)\n")
    
    f.write("\nEarly Payment Patterns:\n")
    f.write("-" * 30 + "\n")
    for (tenure, size), data in early_patterns.iterrows():
        f.write(f"\n{tenure} tenure, {size} loans:\n")
        f.write(f"- Count: {data['days_to_due']['count']:,} loans\n")
        f.write(f"- Average days early: {abs(data['days_to_due']['mean']):.1f} days\n")
        f.write(f"- Earliest payment: {abs(data['days_to_due']['min'])} days before due date\n")
        f.write(f"- Average loan amount: ${data['loan_amount']['mean']:,.2f}\n")
        f.write(f"- Automatic payments: {data[('repayment_type', '<lambda>')]:.1f}%\n")

# Enhanced loan size and payment pattern analysis
print("\n=== LOAN SIZE AND PAYMENT PATTERN ANALYSIS ===")
print("=" * 50)

# Analyze early payment percentage by loan size
size_payment_analysis = merged_df.groupby('loan_size_category').agg({
    'loan_amount': ['count', 'mean', 'sum'],
    'days_to_due': ['mean', 'std'],
    'repayment_type': lambda x: (x == 'Automatic').mean() * 100
}).round(2)

print("\nPayment Patterns by Loan Size:")
print(size_payment_analysis)

# Analyze automatic vs manual payment patterns
payment_type_analysis = merged_df.groupby(['loan_size_category', 'repayment_type']).agg({
    'loan_amount': ['count', 'mean'],
    'days_to_due': 'mean'
}).round(2)

print("\nDetailed Payment Type Analysis:")
print(payment_type_analysis)

# Create visualization for loan size vs payment patterns
plt.figure(figsize=(15, 8))

# Plot 1: Early Payment % by Loan Size
plt.subplot(2, 2, 1)
early_pct_by_size = merged_df.groupby('loan_size_category').apply(
    lambda x: (x['days_to_due'] < 0).mean() * 100
).sort_values()
plt.bar(early_pct_by_size.index, early_pct_by_size.values)
plt.title('Early Payment % by Loan Size')
plt.ylabel('% Early Payments')
plt.xticks(rotation=45)

# Plot 2: Average Days to Due by Size and Payment Type
plt.subplot(2, 2, 2)
avg_days = payment_type_analysis['days_to_due']['mean'].unstack()
avg_days.plot(kind='bar')
plt.title('Average Days to Due by Size and Payment Type')
plt.ylabel('Days')
plt.xticks(rotation=45)

# Plot 3: Payment Type Distribution by Size
plt.subplot(2, 2, 3)
payment_dist = merged_df.groupby('loan_size_category')['repayment_type'].value_counts(normalize=True).unstack()
payment_dist.plot(kind='bar', stacked=True)
plt.title('Payment Type Distribution by Loan Size')
plt.ylabel('Proportion')
plt.xticks(rotation=45)

# Plot 4: Loan Amount vs Days to Due Scatter
plt.subplot(2, 2, 4)
colors = np.where(merged_df['repayment_type'] == 'Automatic', 'blue', 'red')
plt.scatter(merged_df['loan_amount'], merged_df['days_to_due'], alpha=0.1, c=colors)
plt.title('Loan Amount vs Days to Due\n(Blue: Automatic, Red: Manual)')
plt.xlabel('Loan Amount')
plt.ylabel('Days to Due')

plt.tight_layout()
plt.savefig('loan_size_payment_analysis.png', bbox_inches='tight', dpi=300)
print("\nLoan size and payment pattern visualization saved as 'loan_size_payment_analysis.png'")

# Update the report with new analysis
with open('loan_portfolio_report.txt', 'a') as f:
    f.write("\n\nLOAN SIZE AND PAYMENT PATTERN ANALYSIS\n")
    f.write("=" * 50 + "\n")
    
    f.write("\nPayment Patterns by Loan Size:\n")
    for size_cat in size_payment_analysis.index:
        data = size_payment_analysis.loc[size_cat]
        f.write(f"\n{size_cat}:\n")
        f.write(f"- Count: {int(data['loan_amount']['count']):,} loans\n")
        f.write(f"- Average amount: ${float(data['loan_amount']['mean']):,.2f}\n")
        f.write(f"- Total volume: ${float(data['loan_amount']['sum']):,.2f}\n")
        f.write(f"- Average days to due: {float(data['days_to_due']['mean']):,.1f}\n")
        f.write(f"- Payment timing variability: {float(data['days_to_due']['std']):,.1f} days\n")
        f.write(f"- Automatic payment %: {float(data['repayment_type']['<lambda>']):,.1f}%\n")
    
    f.write("\nPayment Type Analysis:\n")
    for (size_cat, payment_type), data in payment_type_analysis.iterrows():
        f.write(f"\n{size_cat} - {payment_type}:\n")
        f.write(f"- Count: {int(data['loan_amount']['count']):,} loans\n")
        f.write(f"- Average amount: ${float(data['loan_amount']['mean']):,.2f}\n")
        f.write(f"- Average days to due: {float(data['days_to_due']):,.1f}\n")

# Detailed 30-day loan analysis
print("\n=== 30-DAY LOAN DETAILED ANALYSIS ===")
print("=" * 50)

# Filter for 30-day loans
loans_30d = merged_df[merged_df['tenure'] == 30]
early_30d = early_payments[early_payments['tenure'] == 30]

# Analyze repayment patterns by amount ranges
loans_30d['amount_range'] = pd.cut(loans_30d['loan_amount'], 
                                 bins=[0, 500, 1500, 2500, 3500], 
                                 labels=['Small (<= 500)', 'Medium (501-1500)', 
                                       'Large (1501-2500)', 'Extra Large (2501+)'])
amount_analysis = loans_30d.groupby('amount_range').agg({
    'loan_amount': ['count', 'mean'],
    'days_to_due': ['mean', 'std'],
    'repayment_type': lambda x: (x == 'Automatic').mean()
}).round(2)

print("\n30-Day Loan Repayment by Amount Range:")
print(amount_analysis)

# Analyze early repayment reasons (based on timing patterns)
very_early = early_30d[early_30d['days_to_due'] < -14]  # More than 2 weeks early
somewhat_early = early_30d[(early_30d['days_to_due'] >= -14) & (early_30d['days_to_due'] < -7)]  # 1-2 weeks early
slightly_early = early_30d[early_30d['days_to_due'] >= -7]  # Up to 1 week early

timing_patterns = pd.DataFrame({
    'category': ['Very Early (>2 weeks)', 'Somewhat Early (1-2 weeks)', 'Slightly Early (<1 week)'],
    'count': [len(very_early), len(somewhat_early), len(slightly_early)],
    'avg_loan': [very_early['loan_amount'].mean(), somewhat_early['loan_amount'].mean(), slightly_early['loan_amount'].mean()],
    'auto_payment_pct': [(very_early['repayment_type'] == 'Automatic').mean() * 100,
                        (somewhat_early['repayment_type'] == 'Automatic').mean() * 100,
                        (slightly_early['repayment_type'] == 'Automatic').mean() * 100]
})

print("\nEarly Repayment Timing Patterns for 30-Day Loans:")
print(timing_patterns.round(2))

# Analyze repeat borrower behavior for 30-day loans
customer_30d_counts = loans_30d.groupby('customer_id').size()
repeat_30d = customer_30d_counts[customer_30d_counts > 1]

print("\nRepeat Borrower Analysis for 30-Day Loans:")
print(f"Total unique customers: {len(customer_30d_counts):,}")
print(f"Repeat customers: {len(repeat_30d):,} ({len(repeat_30d)/len(customer_30d_counts)*100:.1f}%)")
print(f"Average loans per repeat customer: {repeat_30d.mean():.1f}")

# Update the report with 30-day loan analysis
with open('loan_portfolio_report.txt', 'a') as f:
    f.write("\n\n30-DAY LOAN ANALYSIS\n")
    f.write("=" * 50 + "\n")
    
    f.write("\nKey Findings:\n")
    f.write(f"1. Early Repayment Rate: {len(early_30d)/len(loans_30d)*100:.1f}% of 30-day loans\n")
    f.write(f"2. Average Days Early: {abs(early_30d['days_to_due'].mean()):.1f} days\n")
    f.write(f"3. Automatic Payment Usage: {(loans_30d['repayment_type'] == 'Automatic').mean()*100:.1f}%\n")
    
    f.write("\nEarly Repayment Patterns:\n")
    for _, row in timing_patterns.iterrows():
        f.write(f"\n{row['category']}:\n")
        f.write(f"- Count: {row['count']:,} loans\n")
        f.write(f"- Average loan amount: ${row['avg_loan']:,.2f}\n")
        f.write(f"- Automatic payments: {row['auto_payment_pct']:.1f}%\n")
    
    f.write("\nAmount Range Analysis:\n")
    for amount_range in amount_analysis.index:
        data = amount_analysis.loc[amount_range]
        f.write(f"\n{amount_range}:\n")
        f.write(f"- Count: {int(data['loan_amount']['count']):,} loans\n")
        f.write(f"- Average amount: ${float(data['loan_amount']['mean']):,.2f}\n")
        f.write(f"- Average days to repayment: {float(data['days_to_due']['mean']):,.1f} days\n")
        f.write(f"- Payment timing variability: {float(data['days_to_due']['std']):,.1f} days\n")
        f.write(f"- Automatic payments: {float(data['repayment_type']['<lambda>']) * 100:.1f}%\n")

# Create visualization for 30-day loan repayment patterns
plt.figure(figsize=(15, 5))

# Plot 1: Early repayment distribution
plt.subplot(1, 2, 1)
plt.hist(early_30d['days_to_due'].clip(-60, 0), bins=30, edgecolor='black')
plt.title('Early Repayment Distribution (30-day Loans)')
plt.xlabel('Days Early')
plt.ylabel('Number of Loans')

# Plot 2: Amount vs Early Payment Correlation
plt.subplot(1, 2, 2)
plt.scatter(early_30d['loan_amount'], early_30d['days_to_due'], alpha=0.1)
plt.title('Loan Amount vs Early Payment (30-day Loans)')
plt.xlabel('Loan Amount')
plt.ylabel('Days Early')

plt.tight_layout()
plt.savefig('30day_loan_analysis.png', bbox_inches='tight', dpi=300)
print("\n30-day loan analysis visualization saved as '30day_loan_analysis.png'")

# 5. Revenue Analysis
print("\n5. REVENUE ANALYSIS")
print("=" * 50)

daily_metrics = loans_df.groupby('disb_date').agg({
    'loan_fee': 'sum'
}).reset_index()
daily_metrics['moving_avg_revenue'] = daily_metrics['loan_fee'].rolling(window=30).mean()

# Project next 3 months
forecast_daily_revenue = daily_metrics['moving_avg_revenue'].tail(30).mean()
forecast_monthly_revenue = forecast_daily_revenue * 30

print("\nRevenue Forecast (Next 3 Months):")
print(f"Projected Monthly Revenue: ${forecast_monthly_revenue:,.2f}")
print(f"Projected 3-Month Revenue: ${forecast_monthly_revenue * 3:,.2f}")

# Revenue Impact Analysis
print("\n=== REVENUE AND PAYMENT PATTERN IMPACT ===")
print("=" * 50)

# Calculate revenue impact of early payments
early_payment_revenue = merged_df[merged_df['days_to_due'] < 0].agg({
    'loan_amount': 'sum',
    'loan_fee': 'sum'
})

# Calculate effective APR based on actual days held
merged_df['actual_days'] = merged_df['tenure'] + merged_df['days_to_due']
merged_df['effective_apr'] = (merged_df['loan_fee'] / merged_df['loan_amount']) * (365 / merged_df['actual_days']) * 100

# Analyze revenue metrics by payment type
revenue_by_payment = merged_df.groupby('repayment_type').agg({
    'loan_amount': ['count', 'sum'],
    'loan_fee': 'sum',
    'effective_apr': 'mean',
    'actual_days': 'mean'
}).round(2)

print("\nRevenue Impact of Payment Patterns:")
print(revenue_by_payment)

# Analyze manual payment patterns
manual_payments = merged_df[merged_df['repayment_type'] == 'Manual']
manual_early = manual_payments[manual_payments['days_to_due'] < 0]

print("\nManual Payment Analysis:")
print(f"Total manual payments: {len(manual_payments):,}")
print(f"Early manual payments: {len(manual_early):,} ({len(manual_early)/len(manual_payments)*100:.1f}%)")

# Calculate potential revenue impact
avg_fee_per_day = merged_df['loan_fee'] / merged_df['tenure']
early_days_revenue_loss = (manual_early['days_to_due'].abs() * avg_fee_per_day).sum()

# Update the report with comprehensive analysis
with open('loan_portfolio_report.txt', 'a') as f:
    f.write("\n\nREVENUE AND PAYMENT PATTERN ANALYSIS\n")
    f.write("=" * 50 + "\n")
    
    f.write("\n1. Revenue Impact:\n")
    f.write(f"- Early payment volume: ${early_payment_revenue['loan_amount']:,.2f}\n")
    f.write(f"- Early payment fees: ${early_payment_revenue['loan_fee']:,.2f}\n")
    f.write(f"- Estimated revenue impact of early payments: ${early_days_revenue_loss:,.2f}\n")
    
    f.write("\n2. Payment Method Performance:\n")
    for payment_type in revenue_by_payment.index:
        data = revenue_by_payment.loc[payment_type]
        f.write(f"\n{payment_type}:\n")
        f.write(f"- Count: {int(data['loan_amount']['count']):,} loans\n")
        f.write(f"- Total volume: ${float(data['loan_amount']['sum']):,.2f}\n")
        f.write(f"- Total fees: ${float(data['loan_fee']['sum']):,.2f}\n")
        f.write(f"- Average holding period: {float(data['actual_days']):,.1f} days\n")
        f.write(f"- Effective APR: {float(data['effective_apr']):,.1f}%\n")
    
    f.write("\n3. Manual Payment Insights:\n")
    f.write(f"- Early payment rate: {len(manual_early)/len(manual_payments)*100:.1f}%\n")
    f.write(f"- Average days early: {abs(manual_early['days_to_due'].mean()):.1f} days\n")
    
    f.write("\n4. Optimization Strategies:\n")
    f.write("a) Automatic Payment Incentives:\n")
    f.write("   - Consider fee discounts for automatic payment enrollment\n")
    f.write("   - Implement early auto-pay rewards program\n")
    f.write("   - Focus on converting manual payers in Extra Large loan segment\n")
    
    f.write("\nb) Risk-Based Pricing:\n")
    f.write("   - Adjust rates for small loans to reflect higher risk\n")
    f.write("   - Offer better rates to consistent early payers\n")
    f.write("   - Consider tenure-based pricing optimization\n")
    
    f.write("\nc) Product Optimization:\n")
    f.write("   - Introduce flexible payment dates for automatic payments\n")
    f.write("   - Consider shorter tenure options for manual payers\n")
    f.write("   - Develop loyalty program for repeat borrowers\n")

# Create visualization for revenue impact
plt.figure(figsize=(15, 8))

# Plot 1: Revenue Distribution by Payment Type
plt.subplot(2, 2, 1)
revenue_dist = pd.DataFrame({
    'Volume': revenue_by_payment['loan_amount']['sum'],
    'Fees': revenue_by_payment['loan_fee']['sum']
})
revenue_dist.plot(kind='bar')
plt.title('Revenue Distribution by Payment Type')
plt.ylabel('Amount ($)')
plt.xticks(rotation=45)

# Plot 2: Effective APR Distribution
plt.subplot(2, 2, 2)
plt.hist(merged_df['effective_apr'].clip(0, 200), bins=50)
plt.title('Distribution of Effective APR')
plt.xlabel('APR (%)')
plt.ylabel('Count')

# Plot 3: Payment Timing vs Revenue
plt.subplot(2, 2, 3)
plt.scatter(merged_df['days_to_due'], merged_df['loan_fee'], alpha=0.1)
plt.title('Payment Timing vs Revenue')
plt.xlabel('Days to Due')
plt.ylabel('Loan Fee ($)')

# Plot 4: Actual vs Expected Days
plt.subplot(2, 2, 4)
actual_vs_expected = pd.DataFrame({
    'Expected': merged_df['tenure'],
    'Actual': merged_df['actual_days']
}).mean()
actual_vs_expected.plot(kind='bar')
plt.title('Average Expected vs Actual Loan Duration')
plt.ylabel('Days')

plt.tight_layout()
plt.savefig('revenue_impact_analysis.png', bbox_inches='tight', dpi=300)
print("\nRevenue impact visualization saved as 'revenue_impact_analysis.png'")

# Save comprehensive report
with open('loan_portfolio_report.txt', 'w') as f:
    f.write("LOAN PORTFOLIO ANALYSIS REPORT\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. PRODUCT FEATURES\n")
    f.write("-" * 30 + "\n")
    f.write(f"Loan amount range: ${loans_df['loan_amount'].min():,.2f} to ${loans_df['loan_amount'].max():,.2f}\n")
    f.write(f"Average loan amount: ${loans_df['loan_amount'].mean():,.2f}\n")
    f.write(f"Available tenures: {', '.join(map(str, sorted(loans_df['tenure'].unique())))} days\n\n")
    
    f.write("Fee Structure:\n")
    for tenure in sorted(loans_df['tenure'].unique()):
        avg_rate = loans_df[loans_df['tenure'] == tenure]['effective_rate'].mean()
        f.write(f"- {tenure}-day loans: {avg_rate:.2f}% effective interest rate\n")
    
    f.write("\n2. REPAYMENT PATTERNS\n")
    f.write("-" * 30 + "\n")
    f.write("Repayment Types:\n")
    for rtype, count in repayment_types.items():
        f.write(f"- {rtype}: {count:,} ({count/len(repayments_df)*100:.1f}%)\n")
    
    f.write("\nRepayment Timing:\n")
    for tenure in sorted(loans_df['tenure'].unique()):
        timing = merged_df[merged_df['tenure'] == tenure]['days_to_repay']
        f.write(f"- {tenure}-day loans: {timing.mean():.1f} days on average\n")
    
    f.write("\n3. PORTFOLIO METRICS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total loan volume: ${loans_df['loan_amount'].sum():,.2f}\n")
    f.write(f"Total fee revenue: ${loans_df['loan_fee'].sum():,.2f}\n")
    f.write(f"Number of loans: {len(loans_df):,}\n")
    f.write(f"Unique customers: {loans_df['customer_id'].nunique():,}\n")
    f.write(f"Repeat customers: {sum(customer_loans > 1):,}\n")
    
    f.write("\n4. RISK METRICS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total outstanding amount: ${active_loans['loan_amount'].sum():,.2f}\n")
    f.write(f"Number of active loans: {len(active_loans):,}\n")
    f.write("Repayment Performance:\n")
    for size_cat in repayment_by_size.index:
        perf = repayment_by_size.loc[size_cat]
        f.write(f"- {size_cat} loans: {perf['days_to_repay']['mean']:.1f} days average repayment time\n")
    
    f.write("\n5. REVENUE FORECAST\n")
    f.write("-" * 30 + "\n")
    f.write(f"Projected monthly revenue: ${forecast_monthly_revenue:,.2f}\n")
    f.write(f"3-month revenue forecast: ${forecast_monthly_revenue * 3:,.2f}\n")

print("\nDetailed report saved to 'loan_portfolio_report.txt'")

# Create comprehensive visualization
plt.figure(figsize=(15, 12))

# Plot 1: Repayment Timing Distribution
plt.subplot(2, 2, 1)
plt.hist(merged_df['days_to_repay'].dropna(), bins=30, edgecolor='black')
plt.title('Distribution of Days to Repay')
plt.xlabel('Days')
plt.ylabel('Frequency')

# Plot 2: Monthly Performance
plt.subplot(2, 2, 2)
monthly_volume = monthly_metrics['loan_amount']['sum']
monthly_revenue = monthly_metrics['loan_fee']
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(range(len(monthly_volume)), monthly_volume.values, 'b-', label='Loan Volume')
ax2.plot(range(len(monthly_revenue)), monthly_revenue.values, 'r-', label='Revenue')
ax1.set_title('Monthly Volume and Revenue')
ax1.set_xlabel('Month Index')
ax1.set_ylabel('Loan Volume ($)', color='b')
ax2.set_ylabel('Revenue ($)', color='r')

# Plot 3: Repayment Type Distribution
plt.subplot(2, 2, 3)
plt.pie(repayment_types.values, labels=repayment_types.index, autopct='%1.1f%%')
plt.title('Repayment Type Distribution')

# Plot 4: Average Repayment Time by Loan Size
plt.subplot(2, 2, 4)
avg_repay_time = repayment_by_size['days_to_repay']['mean']
plt.bar(range(len(avg_repay_time)), avg_repay_time.values)
plt.xticks(range(len(avg_repay_time)), avg_repay_time.index, rotation=45)
plt.title('Average Repayment Time by Loan Size')
plt.xlabel('Loan Size Category')
plt.ylabel('Average Days to Repay')

plt.tight_layout()
plt.savefig('loan_portfolio_analysis.png', bbox_inches='tight', dpi=300)
print("Visualization saved to 'loan_portfolio_analysis.png'")
