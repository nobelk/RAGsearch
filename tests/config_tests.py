import importlib
import os
from unittest.mock import mock_open, patch

import pytest


class TestResolveString:
    """Test _resolve: env var > config.yaml > default."""

    def _reload_config(self):
        import app.config as cfg

        importlib.reload(cfg)
        return cfg

    def test_default_value(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("app.config._load_yaml_config", return_value={}):
                from app.config import _resolve

                result = _resolve("embedding_model", "OLLAMA_EMBEDDING_MODEL")
                assert result == "nomic-embed-text"

    def test_yaml_overrides_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "app.config._load_yaml_config",
                return_value={"embedding_model": "custom-model"},
            ):
                from app.config import _resolve

                result = _resolve("embedding_model", "OLLAMA_EMBEDDING_MODEL")
                assert result == "custom-model"

    def test_env_overrides_yaml(self):
        with patch.dict(
            os.environ, {"OLLAMA_EMBEDDING_MODEL": "env-model"}, clear=False
        ):
            with patch(
                "app.config._load_yaml_config",
                return_value={"embedding_model": "yaml-model"},
            ):
                from app.config import _resolve

                result = _resolve("embedding_model", "OLLAMA_EMBEDDING_MODEL")
                assert result == "env-model"

    def test_empty_env_falls_through(self):
        with patch.dict(os.environ, {"OLLAMA_EMBEDDING_MODEL": ""}, clear=False):
            with patch("app.config._load_yaml_config", return_value={}):
                from app.config import _resolve

                result = _resolve("embedding_model", "OLLAMA_EMBEDDING_MODEL")
                assert result == "nomic-embed-text"


class TestResolveBool:
    """Test _resolve_bool with various input types."""

    def test_default_true(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("app.config._load_yaml_config", return_value={}):
                from app.config import _resolve_bool

                result = _resolve_bool(
                    "classifier_enabled", "APP_CLASSIFIER_ENABLED"
                )
                assert result is True

    def test_env_false(self):
        with patch.dict(
            os.environ, {"APP_CLASSIFIER_ENABLED": "false"}, clear=False
        ):
            from app.config import _resolve_bool

            result = _resolve_bool("classifier_enabled", "APP_CLASSIFIER_ENABLED")
            assert result is False

    def test_env_true_variants(self):
        from app.config import _resolve_bool

        for value in ("true", "1", "yes", "TRUE", "Yes"):
            with patch.dict(
                os.environ, {"APP_CLASSIFIER_ENABLED": value}, clear=False
            ):
                result = _resolve_bool(
                    "classifier_enabled", "APP_CLASSIFIER_ENABLED"
                )
                assert result is True, f"Expected True for env={value!r}"

    def test_yaml_bool_native(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "app.config._load_yaml_config",
                return_value={"classifier_enabled": False},
            ):
                from app.config import _resolve_bool

                result = _resolve_bool(
                    "classifier_enabled", "APP_CLASSIFIER_ENABLED"
                )
                assert result is False

    def test_yaml_string_bool(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "app.config._load_yaml_config",
                return_value={"classifier_enabled": "no"},
            ):
                from app.config import _resolve_bool

                result = _resolve_bool(
                    "classifier_enabled", "APP_CLASSIFIER_ENABLED"
                )
                assert result is False


class TestLoadYamlConfig:
    """Test _load_yaml_config edge cases."""

    def test_missing_file_returns_empty(self):
        with patch("app.config.Path.cwd") as mock_cwd:
            from pathlib import Path

            mock_cwd.return_value = Path("/nonexistent")
            from app.config import _load_yaml_config

            result = _load_yaml_config()
            assert result == {}

    def test_non_dict_yaml_returns_empty(self):
        from app.config import _load_yaml_config

        with patch("app.config.Path.cwd") as mock_cwd:
            from pathlib import Path

            tmp = Path("/tmp/test_config_yaml")
            tmp.mkdir(exist_ok=True)
            config_file = tmp / "config.yaml"
            config_file.write_text("just a string\n")
            mock_cwd.return_value = tmp
            try:
                result = _load_yaml_config()
                assert result == {}
            finally:
                config_file.unlink()
                tmp.rmdir()
