
CREATE TABLE IF NOT EXISTS retail_transactions (
    transaction_id BIGSERIAL PRIMARY KEY,
    invoice TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    description TEXT,
    quantity INTEGER NOT NULL,
    invoice_date TIMESTAMP NOT NULL,
    price NUMERIC(18,4) NOT NULL,
    customer_id TEXT,
    country TEXT,
    revenue NUMERIC(20,4) NOT NULL,

    CONSTRAINT positive_quantity CHECK (quantity > 0),
    CONSTRAINT positive_price CHECK (price > 0),
    CONSTRAINT positive_revenue CHECK (revenue > 0)
);

CREATE INDEX IF NOT EXISTS idx_retail_invoice_date
ON retail_transactions(invoice_date);

CREATE INDEX IF NOT EXISTS idx_retail_customer
ON retail_transactions(customer_id);

CREATE INDEX IF NOT EXISTS idx_retail_stock_code
ON retail_transactions(stock_code);
