-- Optional: Indexes to speed up window function
CREATE INDEX IF NOT EXISTS idx_customers_sort ON customers(event_type, product_id, event_time);

-- Remove near-duplicates based on 1-second rule
DELETE FROM customers
WHERE ctid IN (
  SELECT ctid FROM (
    SELECT
      ctid,
      ROW_NUMBER() OVER (
        PARTITION BY event_type, product_id
        ORDER BY event_time
      ) AS row_num,
      LAG(event_time) OVER (
        PARTITION BY event_type, product_id
        ORDER BY event_time
      ) AS prev_time
    FROM customers
  ) sub
  WHERE row_num > 1
    AND EXTRACT(EPOCH FROM event_time - prev_time) <= 1
);
