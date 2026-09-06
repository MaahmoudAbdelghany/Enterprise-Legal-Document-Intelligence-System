"""Unit tests for centralized configuration (app/config.py)."""

import os
from unittest.mock import patch
from app.config import Settings, get_settings


def test_default_settings():
    """Verify default settings load correctly without environment variables."""
    settings = Settings(
        _env_file=None,  # Do not read .env file for pure default testing
    )
    assert settings.app_name == "LexisGraph"
    assert settings.app_env == "development"
    assert settings.debug is True
    assert settings.port == 8000
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.groq_vision_model == "llama-3.2-11b-vision-preview"
    assert settings.qdrant_port == 6333
    assert settings.qdrant_vector_dim == 128
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.colpali_model_name == "vidore/colpali-v1.3-hf"
    assert settings.colpali_dpi == 200
    assert settings.presidio_default_lang == "ar"
    assert settings.bm25_top_k == 10
    assert settings.rrf_k_constant == 60
    assert settings.storage_mode == "local"


def test_custom_environment_overrides():
    """Verify environment variables properly override default settings."""
    env_vars = {
        "APP_NAME": "LexisGraph-Test",
        "APP_ENV": "production",
        "DEBUG": "false",
        "PORT": "9000",
        "GROQ_API_KEY": "gsk_test_key_12345",
        "QDRANT_USE_HTTPS": "true",
        "QDRANT_PORT": "443",
        "COLPALI_DEVICE": "cpu",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        settings = Settings(_env_file=None)
        assert settings.app_name == "LexisGraph-Test"
        assert settings.app_env == "production"
        assert settings.debug is False
        assert settings.port == 9000
        assert settings.groq_api_key == "gsk_test_key_12345"
        assert settings.has_groq is True
        assert settings.is_production is True
        assert settings.is_development is False
        assert settings.qdrant_use_https is True
        assert settings.qdrant_url == "https://localhost:443"
        assert settings.resolved_device == "cpu"


def test_helper_properties():
    """Verify computed properties like storage_path, has_openai, and is_development."""
    settings = Settings(
        _env_file=None,
        openai_api_key="sk-test123456",
        local_storage_dir="./custom_storage",
    )
    assert settings.has_openai is True
    assert settings.is_development is True
    assert str(settings.storage_path) == "custom_storage"


def test_resolved_device_fallback():
    """Verify resolved_device safely falls back to cpu when auto is specified."""
    settings = Settings(_env_file=None, colpali_device="auto")
    # Device should be either cuda or cpu, without throwing an unhandled exception
    assert settings.resolved_device in ["cuda", "cpu"]


def test_resolved_device_with_mocked_gpu():
    """Verify resolved_device returns cuda when nvidia-smi and torch reports True."""
    with patch("shutil.which", return_value="C:\\Windows\\System32\\nvidia-smi.exe"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "True\n"
        settings = Settings(_env_file=None, colpali_device="auto")
        assert settings.resolved_device == "cuda"


def test_get_settings_cached():
    """Verify get_settings returns a singleton cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

