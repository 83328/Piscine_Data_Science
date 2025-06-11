DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS items;

CREATE TABLE customers (
    event_time TIMESTAMP,
    event_type TEXT,
    product_id BIGINT,
    price NUMERIC,
    user_id BIGINT,
    session_id UUID
);

-- -- Recreate the items table with 4 columns (no price)
CREATE TABLE items (
    product_id BIGINT,
    category_id BIGINT,
    category_code TEXT,
    brand TEXT
);


\copy customers(event_time, event_type, product_id, price, user_id, session_id) FROM '/Data/customer/data_2022_oct.csv' DELIMITER ',' CSV HEADER;
\copy customers(event_time, event_type, product_id, price, user_id, session_id) FROM '/Data/customer/data_2022_nov.csv' DELIMITER ',' CSV HEADER;
\copy customers(event_time, event_type, product_id, price, user_id, session_id) FROM '/Data/customer/data_2022_dec.csv' DELIMITER ',' CSV HEADER;
\copy customers(event_time, event_type, product_id, price, user_id, session_id) FROM '/Data/customer/data_2023_jan.csv' DELIMITER ',' CSV HEADER;
\copy customers(event_time, event_type, product_id, price, user_id, session_id) FROM '/Data/customer/data_2023_feb.csv' DELIMITER ',' CSV HEADER;
\copy items(product_id, category_id, category_code, brand) FROM '/Data/item/item.csv' DELIMITER ',' CSV HEADER;
