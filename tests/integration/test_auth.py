import pytest
from security.auth import create_token, verify_token


class TestAuth:

    def test_create_token_returns_string(self):
        token = create_token({"sub": "user_123"})
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self):
        token = create_token({"sub": "user_123"})
        payload = verify_token(token)
        assert payload["sub"] == "user_123"

    def test_verify_invalid_token_raises(self):
        with pytest.raises(Exception):
            verify_token("this.is.not.a.valid.token")

    def test_verify_tampered_token_raises(self):
        token = create_token({"sub": "user_123"})
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(Exception):
            verify_token(tampered)

    def test_token_endpoint_returns_token(self, test_client):
        response = test_client.post(
            "/token",
            data={"username": "testuser", "password": "testpass"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_token_payload_contains_sub(self):
        token = create_token({"sub": "alice", "role": "admin"})
        payload = verify_token(token)
        assert payload["sub"] == "alice"

    def test_different_users_get_different_tokens(self):
        token_a = create_token({"sub": "user_a"})
        token_b = create_token({"sub": "user_b"})
        assert token_a != token_b
