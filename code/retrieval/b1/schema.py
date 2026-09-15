"""Frozen B1 JSON Schema. Structural validation only; business rules are separate."""

from __future__ import annotations

from typing import Any

ROUTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "mode",
        "tasks",
        "candidate_query",
    ],
    "properties": {
        "mode": {
            "type": "string",
            "enum": [
                "direct",
                "confirm_rewrite",
                "clarify",
                "task_limit",
            ],
        },
        "tasks": {
            "type": "array",
            "minItems": 0,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "route",
                    "query",
                ],
                "properties": {
                    "route": {
                        "type": "string",
                        "enum": [
                            "knowledge",
                            "system",
                            "unsupported",
                        ],
                    },
                    "query": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
            },
        },
        "candidate_query": {
            "type": [
                "string",
                "null",
            ]
        },
    },
}

# vLLM 0.8.5 xgrammar compatibility projection only. Do not encode business
# validation here. minItems / maxItems / minLength stay on ROUTER_JSON_SCHEMA.
ROUTER_GUIDED_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "mode",
        "tasks",
        "candidate_query",
    ],
    "properties": {
        "mode": {
            "type": "string",
            "enum": [
                "direct",
                "confirm_rewrite",
                "clarify",
                "task_limit",
            ],
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "route",
                    "query",
                ],
                "properties": {
                    "route": {
                        "type": "string",
                        "enum": [
                            "knowledge",
                            "system",
                            "unsupported",
                        ],
                    },
                    "query": {
                        "type": "string",
                    },
                },
            },
        },
        "candidate_query": {
            "type": [
                "string",
                "null",
            ]
        },
    },
}
