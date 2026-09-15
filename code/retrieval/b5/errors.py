"""B5 fail-closed errors. No user-facing copy is generated here."""


class RerankerError(Exception):
    error_type: str = "reranker_error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.error_type)


class RerankerConfigurationError(RerankerError):
    error_type = "reranker_configuration_error"


class RerankerModelError(RerankerError):
    error_type = "reranker_model_error"


class RerankerInputTooLongError(RerankerError):
    error_type = "reranker_input_too_long"


class RerankerInferenceError(RerankerError):
    error_type = "reranker_inference_error"
