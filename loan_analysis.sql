-- Loan Product Characteristics Analysis

-- 1. Loan Amount Distribution and Fee Structure
SELECT 
    tenure,
    COUNT(*) as loan_count,
    ROUND(AVG(loan_amount), 2) as avg_loan_amount,
    ROUND(MIN(loan_amount), 2) as min_loan_amount,
    ROUND(MAX(loan_amount), 2) as max_loan_amount,
    ROUND(AVG(loan_fee / loan_amount * 100), 2) as effective_interest_rate
FROM loans
GROUP BY tenure
ORDER BY tenure;

-- 2. Daily Loan Volume Trends
SELECT 
    DATE(disb_date) as disbursement_date,
    COUNT(*) as num_loans,
    ROUND(SUM(loan_amount), 2) as total_loan_volume,
    ROUND(SUM(loan_fee), 2) as total_fee_revenue
FROM loans
GROUP BY DATE(disb_date)
ORDER BY disbursement_date;

-- 3. Customer Loan Frequency Analysis
SELECT 
    loans_per_customer,
    COUNT(*) as customer_count
FROM (
    SELECT customer_id, COUNT(*) as loans_per_customer
    FROM loans
    GROUP BY customer_id
) customer_loans
GROUP BY loans_per_customer
ORDER BY loans_per_customer;

-- 4. Repayment Analysis by Type
SELECT 
    repayment_type,
    COUNT(*) as repayment_count,
    ROUND(SUM(amount), 2) as total_repaid,
    ROUND(AVG(amount), 2) as avg_repayment_amount
FROM repayments
GROUP BY repayment_type;

-- 5. Monthly Performance Metrics
SELECT 
    SUBSTR(rep_month, 1, 4) as year,
    SUBSTR(rep_month, 5, 2) as month,
    COUNT(DISTINCT customer_id) as unique_customers,
    COUNT(*) as repayment_count,
    ROUND(SUM(amount), 2) as total_repaid
FROM repayments
GROUP BY SUBSTR(rep_month, 1, 4), SUBSTR(rep_month, 5, 2)
ORDER BY year, month;

-- 6. Risk Analysis: Outstanding Loans
WITH loan_status AS (
    SELECT 
        l.customer_id,
        l.loan_amount,
        l.loan_fee,
        l.disb_date,
        COALESCE(SUM(r.amount), 0) as total_repaid
    FROM loans l
    LEFT JOIN repayments r ON l.customer_id = r.customer_id
    GROUP BY l.customer_id, l.loan_amount, l.loan_fee, l.disb_date
)
SELECT 
    COUNT(*) as outstanding_loans,
    ROUND(SUM(loan_amount - total_repaid), 2) as total_outstanding_amount,
    ROUND(AVG(loan_amount - total_repaid), 2) as avg_outstanding_per_loan
FROM loan_status
WHERE total_repaid < (loan_amount + loan_fee);

-- Create a comprehensive view for visualization
CREATE VIEW IF NOT EXISTS loan_performance_view AS
SELECT 
    l.customer_id,
    l.disb_date,
    l.tenure,
    l.loan_amount,
    l.loan_fee,
    r.date_time as repayment_date,
    r.amount as repayment_amount,
    r.repayment_type,
    JULIANDAY(r.date_time) - JULIANDAY(l.disb_date) as days_to_repayment,
    CASE 
        WHEN l.loan_amount <= 500 THEN 'Small'
        WHEN l.loan_amount <= 1500 THEN 'Medium'
        WHEN l.loan_amount <= 2500 THEN 'Large'
        ELSE 'Extra Large'
    END as loan_size_category,
    CASE
        WHEN JULIANDAY(r.date_time) - JULIANDAY(l.disb_date) < l.tenure THEN 'Early'
        WHEN JULIANDAY(r.date_time) - JULIANDAY(l.disb_date) = l.tenure THEN 'On Time'
        ELSE 'Late'
    END as payment_status,
    strftime('%Y-%m', r.date_time) as repayment_month,
    strftime('%Y-%m', l.disb_date) as disbursement_month
FROM loans l
LEFT JOIN repayments r ON l.customer_id = r.customer_id;
