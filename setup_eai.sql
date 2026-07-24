-- ═══════════════════════════════════════════════════════════════
-- SETUP EXTERNAL ACCESS INTEGRATION (EAI) untuk Snowflake
-- ═══════════════════════════════════════════════════════════════
-- CATATAN PENTING:
-- EAI harus dibuat di DATABASE dan SCHEMA yang SAMA
-- dengan tempat Streamlit app dijalankan!
-- 
-- Jalankan script ini di Snowflake Worksheet sebelum deploy app
-- ═══════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════
-- LANGKAH 1: Cek database/schema saat ini
-- ═══════════════════════════════════════════════════════════════
-- Catat hasilnya! EAI harus di database/schema yang sama

SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE();

-- ═══════════════════════════════════════════════════════════════
-- LANGKAH 2: Buat Network Rule untuk PyPI
-- ═══════════════════════════════════════════════════════════════

-- Hapus yang lama jika ada
DROP NETWORK RULE IF EXISTS pypi_network_rule;

CREATE NETWORK RULE pypi_network_rule
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = (
        'pypi.org:443',
        'pypi.python.org:443',
        'files.pythonhosted.org:443',
        'fastly.pypi.org:443',
        'github.com:443',
        'raw.githubusercontent.com:443'
    );

-- ═══════════════════════════════════════════════════════════════
-- LANGKAH 3: Buat External Access Integration
-- ═══════════════════════════════════════════════════════════════

-- Hapus yang lama jika ada
DROP EXTERNAL ACCESS INTEGRATION IF EXISTS pypi_eai;

CREATE EXTERNAL ACCESS INTEGRATION pypi_eai
    ALLOWED_NETWORK_RULES = (pypi_network_rule)
    ENABLED = TRUE;

-- ═══════════════════════════════════════════════════════════════
-- LANGKAH 4: Grant permissions ke role yang kamu pakai
-- ═══════════════════════════════════════════════════════════════

-- Ganti 'SYSADMIN' dengan role kamu jika berbeda
GRANT USAGE ON INTEGRATION pypi_eai TO ROLE SYSADMIN;
GRANT USAGE ON INTEGRATION pypi_eai TO ROLE PUBLIC;

-- ═══════════════════════════════════════════════════════════════
-- LANGKAH 5: Verifikasi EAI sudah aktif
-- ═══════════════════════════════════════════════════════════════

SHOW EXTERNAL ACCESS INTEGRATIONS;

-- Cek detail
DESCRIBE EXTERNAL ACCESS INTEGRATION pypi_eai;
