"""Experimental identifiers. Not a production B1 contract version."""

EXPERIMENTAL_CONTRACT_ID = "b1-jev-experimental-v0"
JEV_DECISION_API = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_GATEWAY_API = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
JEV_GATEWAY_MODEL = "typesafe-ai/jev"
AI_GATEWAY_API_KEY_ENV = "AI_GATEWAY_API_KEY"
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
TUZI_API_KEY_ENV = "TUZI_API_KEY"
TUZI_MODEL_ENV = "TUZI_MODEL"
TUZI_BASE_URL_ENV = "TUZI_BASE_URL"
TUZI_DEFAULT_BASE_URL = "https://api.tu-zi.com"
TUZI_DEFAULT_MODEL = "gpt-4o-mini"
TUZI_JEV_MODEL = "jev-1.13"
TUZI_JEV_API = "https://api.tu-zi.com/v1/systemone"

JEV_QUESTION_KEYS = ("primary_route",)
JEV_MODE_QUESTION_KEYS = ("knowledge_mode",)
ROUTE_KEYS = ("knowledge", "system", "unsupported")
KNOWLEDGE_MODE_KEYS = (
    "direct",
    "multi",
    "task_limit",
    "confirm_rewrite",
    "clarify",
)

GLICLASS_MODEL_ENV = "B1_GLICLASS_MODEL_PATH"
GLICLASS_DEFAULT_MODEL_PATH = r"D:\Download\gliclass-multilang-mini"
GLICLASS_LOGICAL_ID = "knowledgator/gliclass-multilang-mini"

JEV_MAX_BODY_BYTES = 32 * 1024
JEV_QUESTIONS_VERSION = "b1-jev-questions-v3"
