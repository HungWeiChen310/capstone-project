import os
import sys
import importlib
import logging
from unittest.mock import patch, MagicMock

import pytest

# Ensure src is in path for imports. This should be at the very top if it's needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import the module that will be reloaded
import config as config_module


def reload_config_module():
    """Relods the config module to re-evaluate class variables based on mocked os.environ."""
    return importlib.reload(config_module)


@pytest.fixture(autouse=True)
def reset_config_env_vars():
    """Fixture to ensure a clean environment for each test by resetting relevant env vars."""
    original_env = os.environ.copy()
    env_keys_to_manage = [
        "APP_ENV",
        "TESTING",
        "FLASK_DEBUG",
        "PORT",
        "VALIDATION_MODE",
        "OPENAI_API_KEY",
        "LINE_CHANNEL_ACCESS_TOKEN",
        "LINE_CHANNEL_SECRET",
        "ADMIN_USERNAME",
        "ADMIN_PASSWORD",
        "SECRET_KEY",
        "DB_SERVER",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]
    # Clear these specific keys before each test
    for key in env_keys_to_manage:
        if key in os.environ:
            del os.environ[key]

    yield # Test runs here

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# --- APP_ENV and Dependent Defaults Tests ---

def test_app_env_production():
    with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
        Config = reload_config_module().Config
        assert Config.APP_ENV == "production"
        assert Config.DEBUG is False  # Default for prod
        assert Config.VALIDATION_MODE == "strict"
        assert Config.LOG_LEVEL == logging.INFO
        # Port might be set by getenv("PORT", 443) - check specific value if needed

def test_app_env_development_default():
    # No APP_ENV set, should default to development
    with patch.dict(os.environ, {}, clear=True): # Ensure APP_ENV is not set
        Config = reload_config_module().Config
        assert Config.APP_ENV == "development"
        assert Config.DEBUG is True # Default for dev
        assert Config.VALIDATION_MODE == "loose"
        assert Config.LOG_LEVEL == logging.DEBUG

def test_app_env_development_explicit():
    with patch.dict(os.environ, {"APP_ENV": "development"}, clear=True):
        Config = reload_config_module().Config
        assert Config.APP_ENV == "development"
        assert Config.DEBUG is True
        assert Config.VALIDATION_MODE == "loose"
        assert Config.LOG_LEVEL == logging.DEBUG

def test_app_env_testing_explicit():
    with patch.dict(os.environ, {"APP_ENV": "testing"}, clear=True):
        Config = reload_config_module().Config
        assert Config.APP_ENV == "testing"
        assert Config.DEBUG is True
        assert Config.VALIDATION_MODE == "loose"
        assert Config.LOG_LEVEL == logging.DEBUG

def test_testing_env_variable_overrides_app_env():
    # Legacy TESTING=True should force APP_ENV to "testing"
    with patch.dict(os.environ, {"APP_ENV": "production", "TESTING": "True"}, clear=True):
        Config = reload_config_module().Config
        assert Config.APP_ENV == "testing"
        assert Config.DEBUG is True
        assert Config.VALIDATION_MODE == "loose"

def test_flask_debug_override():
    with patch.dict(os.environ, {"APP_ENV": "production", "FLASK_DEBUG": "True"}, clear=True):
        Config = reload_config_module().Config
        assert Config.APP_ENV == "production" # APP_ENV itself is not changed
        assert Config.DEBUG is True # FLASK_DEBUG overrides the APP_ENV default for DEBUG

# --- Config.validate() Tests ---

def get_valid_mock_env():
    """Helper to get a dictionary of all critical env vars set to valid values."""
    return {
        "OPENAI_API_KEY": "test_openai_key",
        "LINE_CHANNEL_ACCESS_TOKEN": "test_line_token",
        "LINE_CHANNEL_SECRET": "test_line_secret",
        "ADMIN_USERNAME": "admin_user",
        "ADMIN_PASSWORD": "admin_pass",
        "SECRET_KEY": "test_secret_key",
        "DB_SERVER": "test_db_server",
        "DB_NAME": "test_db_name",
        # DB_USER and DB_PASSWORD are not strictly critical with Trusted_Connection=yes
    }

def test_validate_success():
    with patch.dict(os.environ, get_valid_mock_env(), clear=True):
        Config = reload_config_module().Config
        assert Config.validate(exit_on_failure=False) is True # Should not raise error

def test_validate_missing_admin_username():
    env = get_valid_mock_env()
    del env["ADMIN_USERNAME"]
    with patch.dict(os.environ, env, clear=True):
        Config = reload_config_module().Config
        with pytest.raises(ValueError) as excinfo:
            Config.validate(exit_on_failure=False)
        assert "ADMIN_USERNAME" in str(excinfo.value)

def test_validate_missing_admin_password():
    env = get_valid_mock_env()
    del env["ADMIN_PASSWORD"]
    with patch.dict(os.environ, env, clear=True):
        Config = reload_config_module().Config
        with pytest.raises(ValueError) as excinfo:
            Config.validate(exit_on_failure=False)
        assert "ADMIN_PASSWORD" in str(excinfo.value)

def test_validate_missing_secret_key():
    env = get_valid_mock_env()
    del env["SECRET_KEY"]
    with patch.dict(os.environ, env, clear=True):
        Config = reload_config_module().Config
        with pytest.raises(ValueError) as excinfo:
            Config.validate(exit_on_failure=False)
        assert "SECRET_KEY" in str(excinfo.value)

def test_validate_missing_openai_key():
    env = get_valid_mock_env()
    del env["OPENAI_API_KEY"]
    with patch.dict(os.environ, env, clear=True):
        Config = reload_config_module().Config
        with pytest.raises(ValueError) as excinfo:
            Config.validate(exit_on_failure=False)
        assert "OPENAI_API_KEY" in str(excinfo.value)

@patch("sys.exit") # Mock sys.exit
def test_validate_exit_on_failure_true(mock_sys_exit: MagicMock):
    env = get_valid_mock_env()
    del env["OPENAI_API_KEY"] # Make it fail
    with patch.dict(os.environ, env, clear=True):
        Config = reload_config_module().Config
        Config.validate(exit_on_failure=True) # This is the default for validate()
        mock_sys_exit.assert_called_once_with(1)

@patch("sys.exit")
def test_validate_script_level_call_production_exits(mock_sys_exit: MagicMock):
    # Simulate APP_ENV=production, where initial validation should exit on failure
    env = get_valid_mock_env()
    del env["SECRET_KEY"] # A critical var
    with patch.dict(os.environ, {"APP_ENV": "production", **env}, clear=True):
        # The reload itself will trigger the script-level validation
        # which for production, calls validate(exit_on_failure=True)
        importlib.reload(config_module)
        mock_sys_exit.assert_called_once_with(1)


def test_validate_script_level_call_development_logs_warning_and_raises():
    # Simulate APP_ENV=development, initial validation logs warning and raises ValueError
    # (which is caught and logged as warning at script end in config.py)
    env = get_valid_mock_env()
    del env["SECRET_KEY"]

    # We need to capture logs for this test
    # Also, the script-level code in config.py catches the ValueError.
    # This test verifies that Config.validate(exit_on_failure=False) is effectively called
    # and would raise ValueError if not caught by the script-end try-except.
    with patch.dict(os.environ, {"APP_ENV": "development", **env}, clear=True):
        Config = reload_config_module().Config # Reload to apply APP_ENV
        # The script-end logic in config.py calls Config.validate(exit_on_failure=False)
        # for development. We test that this call indeed raises ValueError.
        with pytest.raises(ValueError) as excinfo:
             Config.validate(exit_on_failure=False) # Simulating the effect of loose validation
        assert "SECRET_KEY" in str(excinfo.value)
        # We can't easily test the logger.warning from the script-end try-except here
        # without more complex log capturing setup for the module's load-time code.
        # But we've confirmed the underlying validate() call raises as expected for loose mode.

# Example of how Config object behaves after reload
def test_config_attributes_after_reload():
    with patch.dict(os.environ, {"APP_ENV": "production", "FLASK_DEBUG": "True"}, clear=True):
        # importlib.reload(config_module) # This line reloads and applies the new env vars
        # Config = config_module.Config # Access the reloaded Config
        Config = reload_config_module().Config
        assert Config.APP_ENV == "production"
        assert Config.DEBUG is True # FLASK_DEBUG overrides
        assert Config.VALIDATION_MODE == "strict"

    with patch.dict(os.environ, {"APP_ENV": "development"}, clear=True):
        # importlib.reload(config_module)
        # Config = config_module.Config
        Config = reload_config_module().Config
        assert Config.APP_ENV == "development"
        assert Config.DEBUG is True
        assert Config.VALIDATION_MODE == "loose"
