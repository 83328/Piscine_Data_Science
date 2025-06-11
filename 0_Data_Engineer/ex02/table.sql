CREATE TABLE IF NOT EXISTS data_2022_oct (
    event_time     TIMESTAMP WITH TIME ZONE,   -- from '2022-10-01 00:00:00 UTC'
    event_type     VARCHAR(16),                -- e.g., 'cart', 'view', etc.
    product_id     INTEGER,                    -- numeric ID
    price          NUMERIC(10, 2),             -- precise decimal
    user_id        BIGINT,                     -- larger numeric ID
    user_session   UUID                        -- UUID format
);
