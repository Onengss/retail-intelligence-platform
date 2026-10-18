
-- ==================================================
-- RETAIL INTELLIGENCE - BUSINESS SQL ANALYTICS
-- ==================================================

-- 1. Overall merchandise KPIs
SELECT
    COUNT(*) AS transaction_lines,
    COUNT(DISTINCT invoice) AS total_orders,
    COUNT(DISTINCT customer_id) AS known_customers,
    ROUND(SUM(revenue), 2) AS total_revenue,
    SUM(quantity) AS total_units_sold,
    ROUND(
        SUM(revenue) / NULLIF(COUNT(DISTINCT invoice), 0),
        2
    ) AS average_order_value
FROM retail_transactions;


-- 2. Monthly revenue and month-over-month growth
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', invoice_date)::date
            AS sales_month,
        SUM(revenue) AS monthly_revenue,
        COUNT(DISTINCT invoice) AS total_orders
    FROM retail_transactions
    GROUP BY 1
),
monthly_growth AS (
    SELECT
        sales_month,
        monthly_revenue,
        total_orders,
        LAG(monthly_revenue) OVER (
            ORDER BY sales_month
        ) AS previous_month_revenue
    FROM monthly_sales
)
SELECT
    sales_month,
    ROUND(monthly_revenue, 2) AS revenue,
    total_orders,
    ROUND(
        100 * (
            monthly_revenue - previous_month_revenue
        ) / NULLIF(previous_month_revenue, 0),
        2
    ) AS mom_growth_pct
FROM monthly_growth
ORDER BY sales_month;


-- 3. Top 10 products by merchandise revenue
SELECT
    stock_code,
    MAX(description) AS product_description,
    SUM(quantity) AS units_sold,
    ROUND(SUM(revenue), 2) AS total_revenue
FROM retail_transactions
GROUP BY stock_code
ORDER BY total_revenue DESC
LIMIT 10;


-- 4. Top 10 countries by merchandise revenue
SELECT
    country,
    COUNT(DISTINCT invoice) AS total_orders,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(
        100 * SUM(revenue)
        / NULLIF(
            (SELECT SUM(revenue)
             FROM retail_transactions), 0
        ),
        2
    ) AS revenue_share_pct
FROM retail_transactions
GROUP BY country
ORDER BY total_revenue DESC
LIMIT 10;


-- 5. Customer purchasing behavior
WITH customer_orders AS (
    SELECT
        customer_id,
        COUNT(DISTINCT invoice) AS total_orders,
        SUM(revenue) AS total_spent,
        MAX(invoice_date) AS last_purchase
    FROM retail_transactions
    WHERE customer_id IS NOT NULL
      AND TRIM(customer_id) <> ''
    GROUP BY customer_id
)
SELECT
    COUNT(*) AS total_customers,
    COUNT(*) FILTER (
        WHERE total_orders > 1
    ) AS repeat_customers,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE total_orders > 1
        ) / NULLIF(COUNT(*), 0),
        2
    ) AS repeat_customer_pct,
    ROUND(AVG(total_orders), 2)
        AS avg_orders_per_customer,
    ROUND(AVG(total_spent), 2)
        AS avg_customer_spend
FROM customer_orders;
