import re
from unittest.mock import patch
from security.guardrails import anonymize_pii_input, anonymize_pii_output


EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
CARD_PATTERN  = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")


def has_pii(text: str) -> bool:
    return bool(
        EMAIL_PATTERN.search(text)
        or PHONE_PATTERN.search(text)
        or CARD_PATTERN.search(text)
    )


class TestPIILeakageInput:

    def test_email_stripped_from_input(self):
        result = anonymize_pii_input("Contact me at john.doe@company.com please")
        assert not EMAIL_PATTERN.search(result)

    def test_phone_stripped_from_input(self):
        result = anonymize_pii_input("My phone is 415-555-1234, call anytime")
        assert not PHONE_PATTERN.search(result)

    def test_multiple_pii_types_all_stripped(self):
        text = "Email john@acme.com or call 555-123-4567"
        result = anonymize_pii_input(text)
        assert not has_pii(result)

    def test_no_pii_text_unchanged_in_meaning(self):
        text = "What is the refund policy for electronics?"
        result = anonymize_pii_input(text)
        assert "refund" in result
        assert "electronics" in result


class TestPIILeakageOutput:

    def test_email_stripped_from_output(self):
        result = anonymize_pii_output("The account owner is alice@example.com")
        assert not EMAIL_PATTERN.search(result)

    def test_credit_card_stripped_from_output(self):
        result = anonymize_pii_output("Charged card 4111-1111-1111-1111 successfully")
        assert not CARD_PATTERN.search(result)

    def test_phone_stripped_from_output(self):
        result = anonymize_pii_output("Support line is 800-555-9876")
        assert not PHONE_PATTERN.search(result)

    def test_pii_in_csv_response_stripped(self):
        # Simulates what SQL agent might return from employees.csv
        csv_response = "Employee alice@company.com earns $75,000"
        result = anonymize_pii_output(csv_response)
        assert not EMAIL_PATTERN.search(result)

    def test_no_pii_output_passes_through(self):
        text = "Q3 total revenue was $2.4 million across all regions."
        result = anonymize_pii_output(text)
        assert "Q3" in result
        assert "revenue" in result


class TestPIILeakageEndToEnd:

    @patch("agents.supervisor.run_supervisor")
    def test_api_response_has_no_pii(self, mock_supervisor, test_client, auth_headers):
        # Even if supervisor leaks PII, output guardrail should strip it
        mock_supervisor.return_value = {
            "answer": "The user's email is secret@private.com and phone 999-888-7777",
            "agent_used": "sql",
            "sources": [],
            "tokens_used": 100
        }

        response = test_client.post(
            "/chat",
            json={"message": "Show employee contacts", "session_id": "s1", "user_id": "u1"},
            headers=auth_headers
        )

        assert response.status_code == 200
        answer = response.json()["answer"]
        assert not has_pii(answer), f"PII leaked in response: {answer}"
