
import os

# Reference constants
RECORD = 1.9636454840813407
RECORD_C = RECORD / 2.0

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDS_DIR = os.path.join(BASE_DIR, "records")
CERTIFIED_DIR = os.path.join(RECORDS_DIR, "certified")
LEDGER_PATH = os.path.join(RECORDS_DIR, "ledger.jsonl")
TOPK_PATH = os.path.join(RECORDS_DIR, "top_k.json")
BEST_PATH = os.path.join(BASE_DIR, "best_found.json")
LOG_PATH = os.path.join(BASE_DIR, "search_log.jsonl")

# Rounding
DEFAULT_ROUNDING = {
    "tau": 6,
    "alphas": 6,
    "cs": 6
}

# Search
AUTO_CERTIFY_THRESHOLD = 5e-5 # Gap to CHO record to trigger candidate status
SAFETY_BUFFER = 1e-10 # Buffering for certified claims

os.makedirs(CERTIFIED_DIR, exist_ok=True)
