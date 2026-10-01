"""Shared route-only probes. Not production gold."""

from retrieval.b1.types import ConversationTurn

SMOKE_CASES = (
    {
        "id": "identity-en",
        "query": "How does NASA-STD-4003A define class S bonding?",
        "history": (),
    },
    {
        "id": "system",
        "query": "你现在接入了哪些类型的技术文档？",
        "history": (),
    },
    {
        "id": "follow-up",
        "query": "那钢呢？",
        "history": (
            ConversationTurn(
                raw_user_query="NASA-STD-5006A 对铝合金焊接有什么要求？",
                canonical_user_query="NASA-STD-5006A 对铝合金焊接有什么要求？",
                assistant_response="...",
            ),
        ),
    },
)
