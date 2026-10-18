
-- RETAIL INTELLIGENCE - RFM CUSTOMER ANALYSIS

DROP VIEW IF EXISTS customer_rfm;

CREATE VIEW customer_rfm AS

WITH snapshot AS (
    SELECT
        MAX(invoice_date)::date + 1 AS snapshot_date
    FROM retail_transactions
),

customer_metrics AS (
    SELECT
        customer_id,
        MAX(invoice_date)::date AS last_purchase_date,
        COUNT(DISTINCT invoice) AS frequency,
        SUM(revenue) AS monetary
    FROM retail_transactions
    WHERE customer_id IS NOT NULL
      AND TRIM(customer_id) <> ''
    GROUP BY customer_id
),

rfm_base AS (
    SELECT
        c.customer_id,
        s.snapshot_date - c.last_purchase_date AS recency,
        c.frequency,
        ROUND(c.monetary, 2) AS monetary
    FROM customer_metrics c
    CROSS JOIN snapshot s
),

rfm_scores AS (
    SELECT
        *,
        NTILE(5) OVER (
            ORDER BY recency DESC, customer_id
        ) AS r_score,

        NTILE(5) OVER (
            ORDER BY frequency ASC, customer_id
        ) AS f_score,

        NTILE(5) OVER (
            ORDER BY monetary ASC, customer_id
        ) AS m_score
    FROM rfm_base
)

SELECT
    customer_id,
    recency,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    r_score + f_score + m_score AS rfm_total,

    CASE
        WHEN r_score >= 4
         AND f_score >= 4
         AND m_score >= 4
            THEN 'Champions'

        WHEN r_score >= 3
         AND f_score >= 3
            THEN 'Loyal Customers'

        WHEN r_score >= 4
         AND f_score <= 2
            THEN 'New or Occasional'

        WHEN r_score <= 2
         AND f_score >= 3
            THEN 'At Risk'

        ELSE 'Other Customers'
    END AS segment

FROM rfm_scores;
