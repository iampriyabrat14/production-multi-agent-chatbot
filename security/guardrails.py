from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from fastapi import HTTPException, status
import re

# presidio engines — loaded once at module level for performance
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# known prompt injection + jailbreak patterns
INJECTION_PATTERNS = [
    # instruction override
    r"ignore (all |previous |above |all previous )?instructions",
    r"disregard (all |previous |above |your )?instructions",
    r"forget (all |previous |above |everything )?instructions",
    r"forget everything (above|before|prior)",
    r"override (previous |all )?instructions",
    # system prompt exfiltration
    r"reveal (your |the )?system prompt",
    r"repeat (your |the )?system prompt",
    r"what (are|were) your instructions",
    r"show (me )?(your |the )?prompt",
    # identity / role override
    r"you are now",
    r"act as (a |an )?",
    r"pretend (you are|to be|you have)",
    r"pretend .{0,20} no restrictions",
    r"you have no restrictions",
    # jailbreak keywords
    r"bypass (your |all )?restrictions",
    r"jailbreak",
    r"do anything now",
    r"dan mode",
    # SQL injection
    r"(;|--)\s*(drop|delete|insert|update|select)\s",
    r"union\s+select",
    r"1\s*=\s*1",
    # XSS
    r"<script[\s>]",
    r"javascript:",
    r"onerror\s*=",
    # SSTI probes
    r"\{\{.{0,20}\}\}",
    r"\$\{.{0,20}\}",
]


def check_input_safety(message: str) -> None:
    """
    Scans user input for prompt injection and jailbreak attempts.
    Raises 400 Bad Request if a threat is detected.
    """
    message_lower = message.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, message_lower):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input contains disallowed content.",
            )


def anonymize_pii_input(message: str) -> str:
    """
    Detects and anonymizes PII in user input before sending to LLM.
    Example: "My card is 4111-1111-1111-1111" → "My card is <CREDIT_CARD>"
    """
    results = analyzer.analyze(
        text=message,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE"],
        language="en",
    )
    if not results:
        return message
    anonymized = anonymizer.anonymize(text=message, analyzer_results=results)
    return anonymized.text


def anonymize_pii_output(response: str) -> str:
    """
    Detects and anonymizes sensitive PII in LLM output before sending to user.
    Does NOT mask PERSON names — employee/customer names from internal data are valid.
    Only masks financial and contact PII that should never appear in responses.
    """
    results = analyzer.analyze(
        text=response,
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE"],
        language="en",
    )
    if not results:
        return response
    anonymized = anonymizer.anonymize(text=response, analyzer_results=results)
    return anonymized.text
