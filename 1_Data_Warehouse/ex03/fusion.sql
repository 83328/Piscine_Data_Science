-- Cleanup duplicates (by product_id)
DELETE FROM items a
USING items b
WHERE a.ctid > b.ctid AND a.product_id = b.product_id;

-- Ensure columns exist in customers
ALTER TABLE customers
  ADD COLUMN IF NOT EXISTS category_id BIGINT,
  ADD COLUMN IF NOT EXISTS category_code TEXT,
  ADD COLUMN IF NOT EXISTS brand TEXT;

-- Now update customers using the items table
UPDATE customers
SET category_id   = i.category_id,
    category_code = i.category_code,
    brand         = i.brand
FROM items i
WHERE customers.product_id = i.product_id;
