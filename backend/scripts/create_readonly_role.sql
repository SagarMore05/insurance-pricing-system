-- Run this script as a PostgreSQL superuser (e.g. postgres) AFTER running migrations.
-- Required only if you want to use the NLP Assistant (LangChain SQL agent).
--
-- Usage:
--   psql -U postgres -d insurance_pricing_db -f scripts/create_readonly_role.sql

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_roles WHERE rolname = 'insurance_readonly'
    ) THEN
        CREATE ROLE insurance_readonly WITH LOGIN PASSWORD 'readonly_pass';
        RAISE NOTICE 'Role insurance_readonly created.';
    ELSE
        RAISE NOTICE 'Role insurance_readonly already exists, skipping creation.';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE insurance_pricing_db TO insurance_readonly;
GRANT USAGE ON SCHEMA public TO insurance_readonly;
GRANT SELECT ON customers, vehicles, policies, predictions, claims TO insurance_readonly;
