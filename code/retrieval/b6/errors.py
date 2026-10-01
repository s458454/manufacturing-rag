"""B6 fail-closed errors. No evidence text is truncated here."""


class RecoveryError(Exception):
    """Base error for hierarchical context recovery."""


class RecoveryConfigurationError(RecoveryError):
    """Startup configuration or budget is unusable."""


class RecoveryInputError(RecoveryError):
    """B5 result violates the Top-5 input contract."""


class RecoveryIdentityError(RecoveryError):
    """matched-Leaf runtime identity guard failed.

    This is not a build-fingerprint check.
    """


class CoreLeafExceedsBudgetError(RecoveryError):
    """The highest-priority leaf alone exceeds the evidence budget."""


class LeafCatalogError(RecoveryError):
    """A6 rows projection cannot be loaded."""
