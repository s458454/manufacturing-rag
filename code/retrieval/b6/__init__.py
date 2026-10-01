"""B6 Hierarchical Context Recovery."""

from retrieval.b6.catalog import CatalogLeaf, LeafCatalog, load_leaf_catalog
from retrieval.b6.constants import (
    GENERATION_CONTEXT_LIMIT,
    GENERATION_TOKENIZER_LOGICAL_ID,
    GENERATION_TOKENIZER_REVISION,
    NOMINAL_EVIDENCE_BUDGET,
    PARENT_COVERAGE_MIN,
    RECOVERY_CONTRACT_VERSION,
)
from retrieval.b6.context import RecoveryContext, load_recovery_context, recover_request
from retrieval.b6.errors import (
    CoreLeafExceedsBudgetError,
    LeafCatalogError,
    RecoveryConfigurationError,
    RecoveryError,
    RecoveryIdentityError,
    RecoveryInputError,
)
from retrieval.b6.recover import recover, union_source_spans
from retrieval.b6.types import RecoveredEvidence, RecoveryResult

__all__ = [
    "GENERATION_CONTEXT_LIMIT",
    "GENERATION_TOKENIZER_LOGICAL_ID",
    "GENERATION_TOKENIZER_REVISION",
    "NOMINAL_EVIDENCE_BUDGET",
    "PARENT_COVERAGE_MIN",
    "RECOVERY_CONTRACT_VERSION",
    "CatalogLeaf",
    "CoreLeafExceedsBudgetError",
    "LeafCatalog",
    "LeafCatalogError",
    "RecoveredEvidence",
    "RecoveryConfigurationError",
    "RecoveryContext",
    "RecoveryError",
    "RecoveryIdentityError",
    "RecoveryInputError",
    "RecoveryResult",
    "load_leaf_catalog",
    "load_recovery_context",
    "recover",
    "recover_request",
    "union_source_spans",
]
