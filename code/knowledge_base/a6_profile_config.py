"""A6.0 analyzer profiling constants.

Temporary Native BM25 profiling only. This is not the official A6 collection
schema. Analyzer remains [PROVISIONAL]. Runtime does not call A2 or A5.
"""

from __future__ import annotations

from typing import Any

EXPECTED_DOCUMENT_COUNT = 17
EXPECTED_EN_DOCUMENT_COUNT = 14
EXPECTED_CHN_DOCUMENT_COUNT = 3
EXPECTED_LEAF_COUNT = 2034
EXPECTED_EN_QUERY_COUNT = 30
EXPECTED_CHN_QUERY_COUNT = 30

VARCHAR_CONTENT_MAX_BYTES = 65535
TOP_K = 20
RECALL_K_VALUES = (5, 20)

DEFAULT_A3_TOKENIZER = "Qwen/Qwen3-Embedding-4B"

# A1 directory names for the frozen ver1 corpus (A0 stem-hash16).
# These are Leaf.document_id values. Runtime does not read source_sha256.
EN_DOCUMENT_IDS: frozenset[str] = frozenset(
    {
        "09_nist_smart_machining_systems_nistir7218-ea9494477b605074",
        "10_nistir7734_machining_measurement_process_planning-01442a27e5af0ad1",
        "13_nasa_std_5006a_welding_requirements-fa229965758a0f0c",
        "14_nasa_std_8739_4a_crimp_wiring-83d2038895dfb4e1",
        "15_nasa_std_4003a_electrical_bonding-5c5271413bd05f61",
        "16_nasa_std_8739_1b_polymeric_bonding-b2fad7832271f2a3",
        "17_nasa_std_6016c_materials_processes-f3e393e5ff100139",
        "18_nasa_std_6030_additive_manufacturing_systems-5246d3b992ac0cc7",
        "19_nasa_std_6033_am_equipment_facility-7e4c13bcf1061520",
        "20_nasa_std_5017b_mechanisms-9fc0210ae0ecc656",
        "21_nasa_std_5009c_nde-9ecc604979b627f5",
        "22_nasa_std_8739_12a_metrology_calibration-d36a7408a908f89d",
        "23_nasa_hdbk_8739_19_2_mte_specifications-c76fbb6d954f8a89",
        "24_nasa_hdbk_8739_19_3_measurement_uncertainty-61c083e859082cf8",
    }
)

CHN_DOCUMENT_IDS: frozenset[str] = frozenset(
    {
        "YH5WR-17_45型避雷器技术条件书-c0c3c9796c3ca7b3",
        "附件1_深圳供电局有限公司人才发展中心2026年水贝培训基地变电站仿真实训室设备购置项目技术规范书-162d87245ef57765",
        "附件1_电动汽车兆瓦级充电设备技术规范书-61e95bb4c9d1c8a5",
    }
)

PROFILE_EN_STANDARD = "EN-STD"
PROFILE_EN_ICU = "EN-ICU"
PROFILE_CHN_CHINESE = "CHN-BUILTIN"
PROFILE_CHN_JIEBA = "CHN-JIEBA"
PROFILE_CHN_ICU = "CHN-ICU"

ANALYZER_PARAMS: dict[str, dict[str, Any]] = {
    PROFILE_EN_STANDARD: {"type": "standard"},
    PROFILE_EN_ICU: {
        "tokenizer": "icu",
        "filter": ["lowercase", "removepunct"],
    },
    PROFILE_CHN_CHINESE: {"type": "chinese"},
    PROFILE_CHN_JIEBA: {
        "tokenizer": "jieba",
        "filter": ["removepunct"],
    },
    PROFILE_CHN_ICU: {
        "tokenizer": "icu",
        "filter": ["lowercase", "removepunct"],
    },
}

EN_PROFILES = (PROFILE_EN_STANDARD, PROFILE_EN_ICU)
CHN_PROFILES = (PROFILE_CHN_CHINESE, PROFILE_CHN_JIEBA, PROFILE_CHN_ICU)
ALL_PROFILES = EN_PROFILES + CHN_PROFILES

BM25_INDEX_TYPE = "SPARSE_INVERTED_INDEX"
BM25_METRIC_TYPE = "BM25"
BM25_INVERTED_INDEX_ALGO = "DAAT_MAXSCORE"
BM25_K1 = 1.2
BM25_B = 0.75

BM25_INDEX_PARAMS: dict[str, Any] = {
    "index_type": BM25_INDEX_TYPE,
    "metric_type": BM25_METRIC_TYPE,
    "params": {
        "inverted_index_algo": BM25_INVERTED_INDEX_ALGO,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
    },
}

COLLECTION_FIELD_CHUNK_ID = "chunk_id"
COLLECTION_FIELD_DOCUMENT_ID = "document_id"
COLLECTION_FIELD_CONTENT = "content"
COLLECTION_FIELD_SPARSE = "sparse_vector"
SEARCH_ANNS_FIELD = COLLECTION_FIELD_SPARSE

CHUNK_ID_MAX_BYTES = 128
DOCUMENT_ID_MAX_BYTES = 512

EN_CATEGORY_MINIMA: dict[str, int] = {
    "technical_term": 6,
    "multiword_phrase": 6,
    "standard_or_model": 4,
    "numeric_unit": 4,
    "section_or_requirement": 4,
    "process_or_operation": 6,
}

CHN_CATEGORY_MINIMA: dict[str, int] = {
    "technical_term": 6,
    "equipment_name": 5,
    "technical_action": 5,
    "model": 4,
    "numeric_unit": 4,
    "section_or_param": 3,
    "mixed_term": 5,
}

CHN_QUERIES_PER_DOC_MIN = 6
CHN_QUERIES_PER_DOC_MAX = 12
EN_MIN_DOCUMENT_COVERAGE = 10

# Tokenizer smoke strings. Token sequences must be measured with run_analyzer;
# composite model numbers must not be pre-judged.
SPOTCHECK_EN: tuple[str, ...] = (
    "electrical bonding for NASA launch vehicles",
    "NASA-STD-4003A",
    "NASA-STD-8739.4A",
    "NISTIR7218",
    "NISTIR 7734",
    "150 V DC",
    "fracture-critical metallic components",
    "Class A (spaceflight)",
    "section 4.1.2 requirements",
    "additive manufacturing equipment and facility control",
)

SPOTCHECK_ZH: tuple[str, ...] = (
    "避雷器额定电压要求",
    "变电站仿真实训室设备购置",
    "YH5WR-17/45",
    "10kV",
    "20kV",
    "兆瓦级充电设备",
    "额定电压17kV",
    "电动汽车充电设备（兆瓦级）",
    "交流10kV/20kV配电",
    "YH5WR-17/45型避雷器",
)

QUERY_META_PURPOSE = "A6.0 analyzer profiling only"

OUTCOME_UNIFIED_ICU = "UNIFIED_ICU"
OUTCOME_LANGUAGE_SPECIFIC = "LANGUAGE_SPECIFIC_REQUIRED"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
