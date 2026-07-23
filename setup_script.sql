-- ═══════════════════════════════════════════════════════════════
-- Snowflake Native App - Setup Script
-- Terjemah Braille ke Latin
-- ═══════════════════════════════════════════════════════════════

-- Buat database untuk menyimpan data aplikasi
CREATE DATABASE IF NOT EXISTS braille_db;
USE DATABASE braille_db;

-- Buat schema untuk aplikasi
CREATE SCHEMA IF NOT EXISTS app_schema;
USE SCHEMA app_schema;

-- Buat stage untuk upload gambar (opsional)
CREATE STAGE IF NOT EXISTS image_stage
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage untuk upload gambar Braille';

-- Buat warehouse untuk menjalankan query
CREATE WAREHOUSE IF NOT EXISTS braille_warehouse
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    COMMENT = 'Warehouse untuk aplikasi Braille';
