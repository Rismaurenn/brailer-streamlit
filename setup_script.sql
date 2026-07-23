-- ═══════════════════════════════════════════════════════════════
-- Snowflake Native App - Setup Script
-- Terjemah Braille ke Latin
-- ═══════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════
-- 1. BUAT DATABASE DAN SCHEMA
-- ═══════════════════════════════════════════════════════════════
CREATE DATABASE IF NOT EXISTS braille_db;
USE DATABASE braille_db;

CREATE SCHEMA IF NOT EXISTS app_schema;
USE SCHEMA app_schema;

-- ═══════════════════════════════════════════════════════════════
-- 2. BUAT WAREHOUSE
-- ═══════════════════════════════════════════════════════════════
CREATE WAREHOUSE IF NOT EXISTS braille_warehouse
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = FALSE
    COMMENT = 'Warehouse untuk aplikasi Braille';

USE WAREHOUSE braille_warehouse;

-- ═══════════════════════════════════════════════════════════════
-- 3. SETUP EXTERNAL ACCESS INTEGRATION (EAI)
--    Agar pip bisa download package dari PyPI
-- ═══════════════════════════════════════════════════════════════

-- 3a. Buat API Integration untuk PyPI
CREATE OR REPLACE API INTEGRATION pypi_api_integration
    API_PROVIDER = PYPY_API_PROVIDER
    ENABLED = TRUE;

-- 3b. Buat Network Rule untuk akses PyPI
CREATE OR REPLACE NETWORK RULE pypi_network_rule
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = (
        'pypi.org:443',
        'pypi.python.org:443',
        'files.pythonhosted.org:443',
        'fastly.pypi.org:443'
    );

-- 3c. Buat External Access Integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION pypi_eai
    ALLOWED_NETWORK_RULES = (pypi_network_rule)
    ENABLED = TRUE;

-- ═══════════════════════════════════════════════════════════════
-- 4. BUAT STAGE UNTUK UPLOAD GAMBAR (opsional)
-- ═══════════════════════════════════════════════════════════════
CREATE STAGE IF NOT EXISTS image_stage
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage untuk upload gambar Braille';

-- ═══════════════════════════════════════════════════════════════
-- 5. GRANT PERMISSIONS
-- ═══════════════════════════════════════════════════════════════
GRANT USAGE ON INTEGRATION pypi_eai TO ROLE PUBLIC;
GRANT USAGE ON INTEGRATION pypi_api_integration TO ROLE PUBLIC;
GRANT USAGE ON WAREHOUSE braille_warehouse TO ROLE PUBLIC;
GRANT ALL ON DATABASE braille_db TO ROLE PUBLIC;
GRANT ALL ON SCHEMA app_schema TO ROLE PUBLIC;
