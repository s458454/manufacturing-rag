"""Experimental errors. Not part of production B1 user-reply table."""


class JevExperimentalError(Exception):
    error_type = "jev_experimental_error"


class JevProtocolError(JevExperimentalError):
    error_type = "jev_protocol_error"


class JevUncertainError(JevExperimentalError):
    """Classifier is unsure. Must not map to clarify / knowledge / unsupported."""

    error_type = "jev_decision_uncertain"


class JevUnavailableError(JevExperimentalError):
    error_type = "jev_unavailable"
