import os
import sys
import pytest
from unittest.mock import patch

# Add the explicit python path in case of import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.powerbi_integration import (
    get_powerbi_access_token,
    get_powerbi_embed_token,
    get_powerbi_embed_config,
)

# 使用 pytest 的 monkeypatch 自動設定環境變數
@pytest.fixture(autouse=True)
def set_powerbi_env(monkeypatch):
    monkeypatch.setenv("POWERBI_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("POWERBI_CLIENT_SECRET", "dummy_client_secret")
    monkeypatch.setenv("POWERBI_TENANT_ID", "dummy_tenant_id")
    monkeypatch.setenv("POWERBI_WORKSPACE_ID", "dummy_workspace_id")
    monkeypatch.setenv("POWERBI_REPORT_ID", "dummy_report_id")


def fake_post_access_token(url, data, **kwargs):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"access_token": "fake_access_token"}

    return FakeResponse()


def fake_post_embed_token(url, json, headers, **kwargs):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"token": "fake_embed_token"}

    return FakeResponse()


@patch("requests.post", side_effect=fake_post_access_token)
def test_get_powerbi_access_token(mock_post):
    token = get_powerbi_access_token()
    assert token == "fake_access_token"
    mock_post.assert_called_once()


@patch("requests.post", side_effect=fake_post_embed_token)
def test_get_powerbi_embed_token(mock_post):
    token = get_powerbi_embed_token("fake_access_token")
    assert token == "fake_embed_token"
    mock_post.assert_called_once()


@patch("src.powerbi_integration.get_powerbi_embed_token", return_value="fake_embed_token")
@patch("src.powerbi_integration.get_powerbi_access_token", return_value="fake_access_token")
def test_get_powerbi_embed_config(mock_access_token, mock_embed_token):
    config = get_powerbi_embed_config()
    expected_embed_url = "https://app.powerbi.com/reportEmbed?reportId=dummy_report_id&groupId=dummy_workspace_id"
    assert config["embedUrl"] == expected_embed_url
    assert config["accessToken"] == "fake_embed_token"
    assert config["reportId"] == "dummy_report_id"
    assert config["workspaceId"] == "dummy_workspace_id"
