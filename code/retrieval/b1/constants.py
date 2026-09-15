"""Frozen B1 versions, context budget, confirmation lexicons, and user replies."""

from __future__ import annotations

ROUTER_CONTRACT_VERSION = "b1-contract-v1"
ROUTER_PROMPT_VERSION = "b1-router-v3"
ROUTER_SCHEMA_VERSION = "b1-schema-v1"

ROUTER_MODEL_LOGICAL_ID = "Qwen/Qwen2.5-7B-Instruct"
STRUCTURED_DECODING_BACKEND = "xgrammar"

ROUTER_CONTEXT_LIMIT = 32768
MAX_ROUTER_MEMORY_TOKENS = 16384
ROUTER_NON_HISTORY_RESERVE = 8192
ROUTER_OUTPUT_AND_SAFETY_RESERVE = 8192
ROUTER_MAX_NEW_TOKENS = 2048

MAX_QUERY_LENGTH = 2000
MAX_ROUTER_TASKS = 8

ROUTER_TEMPERATURE = 0
ROUTER_TOP_P = 1.0
ROUTER_STREAM = False

CONFIRM_WORDS = {
    "是",
    "是的",
    "对",
    "对的",
    "可以",
    "继续",
    "确认",
    "没问题",
    "就这样",
    "对，就这样",
    "可以，继续",
    "yes",
    "y",
    "ok",
    "okay",
}

REJECT_WORDS = {
    "不",
    "不是",
    "不对",
    "否",
    "取消",
    "no",
    "n",
    "cancel",
}

TRAILING_CONFIRMATION_PUNCTUATION = ("。", "！", "!", "？", "?")

USER_REPLY_EMPTY_QUERY = "请输入需要查询的问题。"
USER_REPLY_INVALID_QUERY = "当前请求包含无法处理的输入内容，请修改后重新提交。"
USER_REPLY_QUERY_TOO_LONG = (
    "当前单次问题过长，请将问题控制在约 500 个中文字符或 2000 个英文字符以内后重新提交。"
)
USER_REPLY_CLARIFY = "当前请求表述不清，请明确需求后重试。"
USER_REPLY_TASK_LIMIT = (
    "当前请求包含的独立任务过多，请拆分为每次不超过 8 个独立任务后重试。"
)
USER_REPLY_UNSUPPORTED = "该请求超出了当前助手职责范围，请调整需求后重试。"
USER_REPLY_UNSUPPORTED_SUBTASK = "该子请求超出了当前助手职责范围，未执行。"
USER_REPLY_REWRITE_REJECTED = "请重新说明需要处理的具体问题。"
USER_REPLY_ROUTER_UNAVAILABLE = "当前路由服务暂时不可用，请稍后重试。"
USER_REPLY_ROUTER_PROTOCOL = "当前请求处理出现异常，请重新提交；若仍失败，请稍后重试。"
USER_REPLY_CONFIRM_REWRITE_TEMPLATE = (
    "我将你的请求理解为：“{candidate_query}”。是否按这个请求继续？"
)
