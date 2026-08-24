import ast
import re
import logging
import asyncio
import time
from typing import Optional, Dict, Any, List, Tuple

from sqlalchemy import create_engine, inspect as sa_inspect, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from src.config import settings

logger = logging.getLogger("nlp_assistant")

ALLOWED_TABLES = {"customers", "vehicles", "driving_profiles", "policies", "predictions", "claims"}

# Blocks any DML / DDL keyword anywhere in the SQL string.
BLOCKED_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# Detects common prompt-injection patterns in the user question.
INJECTION_PATTERNS = re.compile(
    r"ignore (previous|all|prior|above) instruction|"
    r"you are now|pretend (to be|you are)|disregard|"
    r"forget (all|previous)|new instructions?|jailbreak|DAN mode",
    re.IGNORECASE,
)

# Explicit singular forms — avoids "policie" from "policies"[:-1]
_SUBJECT_SINGULAR: Dict[str, str] = {
    "policies":         "policy",
    "claims":           "claim",
    "vehicles":         "vehicle",
    "customers":        "customer",
    "predictions":      "prediction",
    "driving_profiles": "driving profile",
    "records":          "record",
}

# Compiled patterns used by _sanitize_answer — defined at module level so they
# are compiled once rather than on every call to _format_answer.
_SQL_CODE_BLOCK = re.compile(r"```(?:sql)?\s*.*?```", re.DOTALL | re.IGNORECASE)
_NOTE_LINE = re.compile(
    r"^\s*(?:"
    r"Note(?:\s+that)?\b|"
    r"Please\s+note\b|"
    r"The\s+query\s+used\b|"
    r"The\s+SQL\s+(?:query\s+)?(?:used|is|was)\b|"
    r"(?:The\s+)?[Aa]ctual\s+output\b"
    r").*$",
    re.MULTILINE | re.IGNORECASE,
)
_META_PREAMBLE = re.compile(
    r"^(?:The query (?:returned|has returned)\s+(?:a |the )?(?:count of\s+)?|"
    r"Based on (?:the )?results?,?\s+)",
    re.IGNORECASE,
)
# Matches Python list-of-tuple SQL echoes: [('hyundai', 1)]
_PYTHON_RESULT = re.compile(r"\[.*\(.*\).*\]")
# Matches SQL-style single-quoted entity names: 'mumbai' → Mumbai
_QUOTED_ENTITY = re.compile(r"'([a-zA-Z][a-zA-Z\s]+)'")
# Strips same-line trailing disclaimer sentences: "...high: 1. Note that..."
# _NOTE_LINE handles full-line disclaimers; this handles the mid-sentence case.
_DISCLAIMER_START = re.compile(
    r"(?<=\.)\s+(?:Note(?:\s+that)?\b|Please\s+note\b|"
    r"The\s+query\s+used\b|The\s+SQL\s+(?:query\s+)?(?:used|is|was)\b|"
    r"(?:The\s+)?[Aa]ctual\s+output\b)",
    re.IGNORECASE,
)

# ── Error classification ──────────────────────────────────────────────────────

_RATE_LIMIT_SIGNALS = re.compile(
    r"429|rate[\s_-]?limit|quota|too\s+many\s+requests|tokens?\s+per\s+(minute|day|hour)",
    re.IGNORECASE,
)
_SQL_ERROR_TYPES = frozenset({
    "OperationalError", "ProgrammingError", "InterfaceError",
    "DatabaseError", "InternalError", "DataError",
})


def _classify_error(exc: Exception) -> str:
    """Returns 'rate_limit', 'sql', or 'ai' — never exposes provider details."""
    if _RATE_LIMIT_SIGNALS.search(str(exc)):
        return "rate_limit"
    exc_type = type(exc).__name__
    exc_str_lower = str(exc).lower()
    if exc_type in _SQL_ERROR_TYPES or "sqlalchemy" in exc_str_lower or "psycopg2" in exc_str_lower:
        return "sql"
    return "ai"


_RETRY_DELAYS: Tuple[int, ...] = (0, 1, 2, 4)  # seconds before each of 4 attempts
_RETRIABLE_EXC_TYPES = frozenset({
    "ConnectionError", "TimeoutError", "ConnectTimeoutError",
    "ReadTimeoutError", "ChunkedEncodingError", "HTTPStatusError",
})


def _is_retriable(exc: Exception) -> bool:
    """True for transient failures worth retrying: rate limits, timeouts, 5xx."""
    # Rate limit / quota — always worth retrying
    if _RATE_LIMIT_SIGNALS.search(str(exc)):
        return True
    exc_type = type(exc).__name__
    # Timeout by exception class name (TimeoutError, ConnectTimeoutError, etc.)
    if "timeout" in exc_type.lower():
        return True
    # Known connection-level exception types
    if exc_type in _RETRIABLE_EXC_TYPES:
        return True
    exc_str = str(exc).lower()
    # Timeout signal anywhere in the message ("timeout" or "timed out")
    if re.search(r"timeout|timed\s+out", exc_str):
        return True
    # Temporary gateway / provider failures (502, 503, 504)
    if re.search(
        r"\b(502|503|504)\b|service[\s_]?unavailable|temporarily[\s_]?unavailable",
        exc_str,
    ):
        return True
    # SQL / DB errors are not transient from the LLM's perspective
    if _classify_error(exc) == "sql":
        return False
    # Default: unknown errors are not retried (conservative)
    return False


# ── System prompt ─────────────────────────────────────────────────────────────
# Key changes vs original:
#   • Explicit sentence-style formatting requirement (fixes bare "1" answers)
#   • [CHART: xxx] marker the LLM appends — parsed out of answer after the call
#     so the LLM's own understanding of result shape drives the suggestion
#   • Explicit SQL rules repeated to reduce hallucination
SYSTEM_PROMPT = """You are a professional insurance analytics copilot for a car insurance business in India.

You have READ-ONLY access to a PostgreSQL database with these tables:
- customers        : customer demographics (customer_id, age, gender, city, created_at, occupation, marital_status, annual_income_band)
- vehicles         : vehicle details (vehicle_id, customer_id, car_brand, car_model, engine_cc, vehicle_age_years, vehicle_value_inr, fuel_type, vehicle_usage_type, parking_type, vehicle_safety_rating, airbags_count, vehicle_body_style, repair_cost_band)
- driving_profiles : driving history and geo-risk (profile_id, customer_id, driving_score, annual_mileage_km, previous_claims_count, years_licensed, months_since_last_claim, policy_tenure_years, no_claim_bonus_pct, region_risk_category, flood_risk_index, theft_risk_index, monsoon_exposure_index, pincode_risk_score, policy_inception_month)
- policies         : insurance policies (policy_id, customer_id, vehicle_id, premium_amount_inr, risk_level, is_active, model_version, created_at)
- predictions      : ML model outputs (prediction_id, policy_id, claim_probability, expected_claim_amount_inr, final_premium_inr)
- claims           : filed claims (claim_id, policy_id, claimed_amount_inr, approved_amount_inr, claim_status, claim_date)

MANDATORY RESPONSE FORMAT — these rules are absolute:
- Your final answer MUST be one or more complete English sentences. Never output a lone number.
- WRONG: "1"   CORRECT: "There is currently 1 active policy in the database."
- WRONG: "152" CORRECT: "There are 152 policies in the database."
- WRONG: "45234.50" CORRECT: "The average premium is ₹45,234.50."
- For counts: start with "There is/are currently N <subject>…"
- For averages: start with "The average <metric> is ₹N." or "The average <metric> is N."
- For totals: start with "The total <metric> is ₹N."
- For rankings/distributions: list results in a sentence, e.g. "High-risk accounts for 25 policies, medium for 78, and low for 49."
- Format currency as Indian Rupees with commas (₹12,45,000). Format percentages to 1 decimal place.
- Keep responses to 1–4 sentences — professional and concise.

TABLE RELATIONSHIPS — use these for JOINs:
- customers.customer_id      = driving_profiles.customer_id (one customer → one driving profile)
- customers.customer_id      = vehicles.customer_id         (one customer → many vehicles)
- customers.customer_id      = policies.customer_id         (one customer → many policies)
- policies.vehicle_id        = vehicles.vehicle_id          (one policy → one vehicle)
- policies.policy_id         = predictions.policy_id        (one policy → one prediction)
- policies.policy_id         = claims.policy_id             (one policy → many claims)

SQL RULES — follow strictly:
- Add LIMIT 100 to every query.
- Only use the 6 tables listed above — no other tables, no schema prefixes.
- Never write UPDATE, INSERT, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
- Never expose exact customer ages — use ranges: CASE WHEN age BETWEEN 18 AND 30 THEN '18-30' …
- When a question names a specific status (pending, approved, rejected), ALWAYS add WHERE claim_status = '<status>' to filter claims.
- When a question asks about "active" policies, ALWAYS add WHERE is_active = true.
- For city comparisons (e.g. "compare Mumbai and Bangalore"), JOIN customers and policies on customer_id and GROUP BY city.
- For questions about driving scores, mileage, or prior claims, JOIN driving_profiles on driving_profiles.customer_id = customers.customer_id.

CHART HINT — on the LAST line of your response, always append one of:
[CHART: bar]   — comparisons, rankings, grouped counts
[CHART: line]  — time series, monthly/yearly trends
[CHART: pie]   — proportions, distributions with ≤ 6 categories
[CHART: none]  — single scalar value, explanations, refusals

If asked to modify data, politely decline and state that you have read-only access."""


# ── Connection diagnostics ────────────────────────────────────────────────────

def _diagnose_connection() -> Dict[str, Any]:
    """
    Validates the DB connection and confirms required tables exist.
    Called once during agent construction — results are always logged.
    Returns a dict; raises nothing (errors surfaced via 'error' key).
    """
    url = settings.DATABASE_URL_SYNC
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            db_name = conn.execute(text("SELECT current_database()")).scalar()
            db_host = conn.execute(text("SELECT inet_server_addr()")).scalar()
        inspector = sa_inspect(engine)
        discovered = sorted(inspector.get_table_names())
        engine.dispose()

        missing = sorted(ALLOWED_TABLES - set(discovered))
        info = {
            "url": url,
            "database": db_name,
            "host": db_host or "localhost",
            "discovered_tables": discovered,
            "include_tables": sorted(ALLOWED_TABLES),
            "missing_from_db": missing,
        }
        logger.info(
            "[NLP-DIAG] database=%s host=%s tables_in_db=%s required=%s missing=%s",
            db_name, info["host"], discovered, sorted(ALLOWED_TABLES), missing,
        )
        return info
    except Exception as exc:
        logger.error("[NLP-DIAG] connection diagnostic failed: %s", exc)
        return {"error": str(exc), "url": url}


# ── SQL / data helpers ────────────────────────────────────────────────────────

def _extract_column_names(sql: str) -> List[str]:
    """
    Parse SELECT column aliases/names from a SQL string.
    Handles: COUNT(*) AS cnt, SUM(x) AS total, p.risk_level, plain col names.
    Falls back to col_0, col_1 … when parsing is ambiguous.
    """
    sql_flat = sql
    while re.search(r'\([^()]*\)', sql_flat):
        sql_flat = re.sub(r'\([^()]*\)', '', sql_flat)
    match = re.search(r"SELECT\s+(.*?)\s+FROM\b", sql_flat, re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    select_part = match.group(1).strip()
    if select_part == "*":
        return []

    # Split by top-level commas only (not inside parentheses)
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in select_part:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())

    names: List[str] = []
    for part in parts:
        as_match = re.search(r"\bAS\s+(\w+)\s*$", part, re.IGNORECASE)
        if as_match:
            names.append(as_match.group(1).lower())
        else:
            # Strip function calls, take last identifier
            bare = re.sub(r"\(.*\)", "", part).strip()
            tokens = re.split(r"[\s.]+", bare)
            names.append(tokens[-1].lower() if tokens and tokens[-1] else f"col_{len(names)}")
    return names


def _parse_sql_observation(raw: str, sql: str) -> Optional[List[Dict[str, Any]]]:
    """
    Convert the string observation returned by the sql_db_query tool into
    a typed list-of-dicts.  Example inputs:
        "[(152,)]"                              → [{"count": 152}]
        "[('high', 25), ('medium', 30)]"        → [{"risk_level": "high", "count": 25}, …]
    Returns None if the string cannot be parsed.
    """
    if not raw or raw.strip() in ("", "None", "[]"):
        return None
    try:
        normalized = re.sub(r"Decimal\('([^']+)'\)", r'\1', raw.strip())
        rows = ast.literal_eval(normalized)
    except Exception:
        return None

    if not isinstance(rows, list) or not rows:
        return None

    col_names = _extract_column_names(sql)
    result: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, (tuple, list)):
            row = (row,)
        if col_names and len(col_names) == len(row):
            result.append({k: v for k, v in zip(col_names, row)})
        else:
            result.append({f"col_{i}": v for i, v in enumerate(row)})
    return result or None


def _suggest_chart_from_result(
    data: Optional[List[Dict[str, Any]]],
    sql: str,
    llm_hint: Optional[str] = None,
) -> str:
    """
    Data-driven chart recommendation. Priority order:
      1. SQL result shape + column names (deterministic)
      2. SQL structure (GROUP BY, ORDER BY, date functions)
      3. LLM hint as tie-breaker only when data analysis is inconclusive
    """
    if not data:
        return "none"
    if len(data) == 1 and len(data[0]) == 1:
        return "none"  # single scalar — no chart needed

    keys = list(data[0].keys())
    num_rows = len(data)
    sql_upper = sql.upper()

    # Time-series detection: column name contains a time-period word
    has_time_col = any(
        re.search(r"\b(month|year|date|time|period|week|quarter|day)\b", k, re.IGNORECASE)
        for k in keys
    )
    if has_time_col and num_rows >= 2:
        return "line"

    # SQL-level date functions → definitely a time series
    if re.search(r"\b(TO_CHAR|DATE_TRUNC|DATE_PART|EXTRACT)\b", sql_upper):
        return "line"

    has_group_by = bool(re.search(r"\bGROUP\s+BY\b", sql_upper))
    if has_group_by:
        # Categorical column with few distinct values → pie
        has_category_col = any(
            re.search(
                r"\b(risk|status|gender|city|brand|type|level|tier|category|class)\b",
                k, re.IGNORECASE,
            )
            for k in keys
        )
        if has_category_col and num_rows <= 6:
            return "pie"
        return "bar"

    # Rankings: ORDER BY + multiple rows → bar comparison
    if re.search(r"\bORDER\s+BY\b", sql_upper) and num_rows > 1:
        return "bar"

    # Multiple rows with at least 2 columns → comparison → bar
    if num_rows > 1 and len(keys) >= 2:
        return "bar"

    # Single row, multiple columns — use LLM hint as tie-breaker
    if llm_hint in ("bar", "line", "pie"):
        return llm_hint

    return "none"


# ── Answer formatter ──────────────────────────────────────────────────────────

def _infer_subject(sql: str) -> str:
    """Guess the entity being counted/aggregated from the SQL."""
    sql_upper = sql.upper()
    if "CLAIMS" in sql_upper:
        return "claims"
    if "POLICIES" in sql_upper:
        return "policies"
    if "DRIVING_PROFILES" in sql_upper:
        return "driving_profiles"
    if "CUSTOMERS" in sql_upper:
        return "customers"
    if "PREDICTIONS" in sql_upper:
        return "predictions"
    if "VEHICLES" in sql_upper:
        return "vehicles"
    return "records"


def _is_currency_query(sql: str) -> bool:
    return bool(re.search(r"premium|amount|value|inr|cost|price", sql, re.IGNORECASE))


def _infer_metric_label(sql: str) -> str:
    sql_lower = sql.lower()
    if re.search(r"\bavg\b|average", sql_lower):
        if "premium" in sql_lower:
            return "average premium"
        if "amount" in sql_lower:
            return "average claim amount"
        return "average value"
    if re.search(r"\bsum\b|total", sql_lower):
        if "premium" in sql_lower:
            return "total premium collected"
        return "total value"
    if re.search(r"\bmax\b", sql_lower):
        return "maximum value"
    if re.search(r"\bmin\b", sql_lower):
        return "minimum value"
    return "value"


def _sanitize_answer(text: str) -> str:
    """
    Strip LLM formatting artifacts before any business-sentence logic runs.
    Removes SQL code fences, markdown table rows, disclaimer/note lines,
    Python tuple/list echoes, SQL-quoted entity names, and the
    'The query returned' meta-preamble.  Safe to call on any string.
    """
    # Remove SQL code fences (```sql ... ```)
    text = _SQL_CODE_BLOCK.sub("", text)
    # Remove markdown table rows — any line that starts with |
    lines = text.splitlines()
    text = "\n".join(line for line in lines if not re.match(r"^\s*\|", line)).strip()
    # Remove disclaimer, note, and query-explanation lines
    text = _NOTE_LINE.sub("", text).strip()
    # Strip same-line trailing disclaimers that follow the answer after a period
    text = _DISCLAIMER_START.split(text, maxsplit=1)[0].strip()
    # Remove lines containing raw Python list-of-tuple output: [('x', 1)]
    lines = text.splitlines()
    text = "\n".join(
        line for line in lines if not _PYTHON_RESULT.search(line)
    ).strip()
    # Collapse runs of blank lines created by the removals above
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Strip "The query returned a count of" / "Based on results," preamble
    text = _META_PREAMBLE.sub("", text)
    # Remove SQL-style single quotes from entity names: 'mumbai' → Mumbai
    text = _QUOTED_ENTITY.sub(lambda m: m.group(1).strip().title(), text)
    # Re-capitalise first character after preamble removal
    if text:
        text = text[0].upper() + text[1:]
    return text


def _format_groupby_sentence(data: List[Dict[str, Any]], sql: str) -> Optional[str]:
    """
    Build a natural English sentence from GROUP BY result data.

    Handles two shapes:
      - Single-column scalar  → "There are/is currently N X in the database."
      - Two-column distribution (2–8 rows) → "Breakdown by X: A: N, B: M, and C: P."

    Returns None for shapes too complex to summarise (>2 cols or >8 rows),
    which fall through to the existing ≥6-word pass-through logic.
    """
    if not data:
        return None

    keys = list(data[0].keys())
    is_currency = _is_currency_query(sql)
    subject = _infer_subject(sql)

    def _fmt(val: Any) -> str:
        try:
            n = float(val)
            if is_currency:
                return f"₹{n:,.2f}"
            return f"{int(n):,}" if n == int(n) else f"{n:,.2f}"
        except (TypeError, ValueError):
            return str(val)

    # Single-column scalar (filtered count that came through a GROUP BY)
    if len(keys) == 1:
        try:
            n = int(float(data[0][keys[0]]))
        except (TypeError, ValueError):
            return None
        singular = _SUBJECT_SINGULAR.get(subject, subject.rstrip("s"))
        word = singular if n == 1 else subject
        verb = "is" if n == 1 else "are"
        return f"There {verb} currently {n:,} {word} in the database."

    # Two-column distribution: label + value, 1–8 rows
    if len(keys) == 2 and 1 <= len(data) <= 8:
        label_key, value_key = keys[0], keys[1]
        parts = [
            f"{str(row[label_key]).capitalize()}: {_fmt(row[value_key])}"
            for row in data
        ]
        group_label = label_key.replace("_", " ")
        if len(parts) == 1:
            joined = parts[0]
        elif len(parts) == 2:
            joined = f"{parts[0]} and {parts[1]}"
        else:
            joined = ", ".join(parts[:-1]) + f", and {parts[-1]}"
        return f"Breakdown by {group_label}: {joined}."

    return None


def _format_trend_summary(data: List[Dict[str, Any]], sql: str) -> Optional[str]:
    """
    Build a concise business summary for time-series (monthly/yearly) data.
    Activates only when data has a recognisable time-period column and ≥2 rows.
    Returns None if the data cannot be summarised simply.
    """
    if not data or len(data) < 2:
        return None

    keys = list(data[0].keys())
    time_key = next(
        (k for k in keys
         if re.search(r"\b(month|year|date|period|week|quarter)\b", k, re.IGNORECASE)),
        None,
    )
    if not time_key:
        return None

    value_keys = [k for k in keys if k != time_key]
    if not value_keys:
        return None

    subject = _infer_subject(sql)
    is_currency = _is_currency_query(sql)
    val_key = value_keys[0]

    try:
        peak_row      = max(data, key=lambda r: float(r[val_key]))
        latest_row    = data[-1]
        peak_period   = str(peak_row[time_key])
        peak_val      = float(peak_row[val_key])
        latest_period = str(latest_row[time_key])
        latest_val    = float(latest_row[val_key])
    except (TypeError, ValueError):
        return None

    def _fmt(v: float) -> str:
        return f"₹{v:,.2f}" if is_currency else f"{int(v):,}"

    n_periods = len(data)
    return (
        f"Trend spans {n_periods} periods. "
        f"Peak was {peak_period} with {_fmt(peak_val)} {subject}; "
        f"the latest period ({latest_period}) had {_fmt(latest_val)}."
    )


def _capitalize_data_values(text: str, data: List[Dict[str, Any]]) -> str:
    """
    Capitalize entity names from query result rows that appear lowercase in text.
    Only applies to purely alphabetic string values (city names, car brands)
    longer than 2 characters.  Numbers, dates, and short tokens are skipped.
    """
    for row in data:
        for val in row.values():
            if (isinstance(val, str)
                    and len(val) > 2
                    and re.match(r"^[a-z][a-z\s]+$", val)):
                text = re.sub(
                    r"\b" + re.escape(val) + r"\b",
                    val.title(),
                    text,
                    flags=re.IGNORECASE,
                )
    return text


def _format_answer(raw: str, data: Optional[List[Dict[str, Any]]], sql: str) -> str:
    """
    Safety-net formatter: converts bare numbers, terse tokens, tabular GROUP BY
    output, and markdown trend tables into professional business sentences using
    the structured data and SQL already available.

    Processing order:
      1. Sanitize markdown tables, SQL fences, Note: lines, meta-preambles
      2. GROUP BY data reconstruction (when output is tabular or short)
      3. Trend summary (when markdown table was stripped, leaving short text)
      4. ≥6-word pass-through (well-formed LLM answer)
      5. Bare-integer promotion
      6. Bare-decimal / currency promotion
      7. Short-phrase punctuation fix
    """
    stripped = _sanitize_answer(raw.strip())
    if data:
        stripped = _capitalize_data_values(stripped, data)

    is_groupby = bool(re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE))
    looks_tabular = "\n" in stripped or bool(
        re.search(r"^\w[\w\s]*:\s*[\d₹]", stripped, re.MULTILINE)
    )

    # Dangling introduction: sanitization removed the data rows but left an
    # intro sentence ending with ":" e.g. "counted as follows:" — reconstruct
    # entirely from structured data rather than returning a broken sentence.
    if data and stripped.endswith(":"):
        sentence = _format_groupby_sentence(data, sql) or _format_trend_summary(data, sql)
        if sentence:
            return sentence

    # GROUP BY reconstruction — fires when output is tabular or fewer than
    # 6 words (not a complete sentence), and structured data is available.
    if data and is_groupby and (looks_tabular or len(stripped.split()) < 6):
        sentence = _format_groupby_sentence(data, sql)
        if sentence:
            return sentence

    # Trend reconstruction — fires when sanitization left fewer than 6 words
    # (e.g. a markdown table was stripped entirely) and data has a time column.
    if data and len(stripped.split()) < 6:
        sentence = _format_trend_summary(data, sql)
        if sentence:
            return sentence

    # Already a proper answer — leave it alone
    if len(stripped.split()) >= 6:
        return stripped

    # Bare integer: "1", "152", "0"
    if re.fullmatch(r"\d+", stripped):
        n = int(stripped)
        subject = _infer_subject(sql)
        singular = _SUBJECT_SINGULAR.get(subject, subject.rstrip("s"))
        if n == 1:
            return f"There is currently {n:,} {singular} in the database."
        return f"There are currently {n:,} {subject} in the database."

    # Bare decimal or currency-looking number: "45234.5", "45,234.50", "1234"
    clean = stripped.replace(",", "").replace("₹", "")
    try:
        val = float(clean)
        metric = _infer_metric_label(sql)
        if _is_currency_query(sql):
            return f"The {metric} is ₹{val:,.2f}."
        return f"The {metric} is {val:,.2f}."
    except ValueError:
        pass

    # Short phrase (1–5 words) that's not a number — ensure it ends with a period
    if not stripped.endswith((".", "!", "?")):
        return stripped + "."

    return stripped


# ── Main assistant class ──────────────────────────────────────────────────────

class InsuranceAssistant:
    def __init__(self):
        self._agent = None
        self._build_error: Optional[str] = None

    def _build_agent(self) -> None:
        """
        Build the LangChain SQL agent.  Called lazily on first query().
        Runs connection diagnostics first so failures are always logged.
        """
        diag = _diagnose_connection()
        if "error" in diag:
            raise ConnectionError(f"Cannot connect to database: {diag['error']}")
        if diag["missing_from_db"]:
            raise ValueError(
                f"Required tables not found in database '{diag['database']}' "
                f"on '{diag['host']}': {diag['missing_from_db']}. "
                f"All tables: {diag['discovered_tables']}"
            )

        db = SQLDatabase.from_uri(
            settings.DATABASE_URL_SYNC,
            include_tables=sorted(ALLOWED_TABLES),
            # sample_rows_in_table_info=0 avoids live SELECT during init;
            # the agent uses sql_db_schema tool at query time instead.
            sample_rows_in_table_info=0,
        )

        from langchain_groq import ChatGroq  # lazy import — avoids hard startup dependency
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=settings.GROQ_API_KEY,
        )

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # return_intermediate_steps=True is the critical flag that makes
        # AgentExecutor.invoke() include the (AgentAction, observation) list in
        # its return dict — without it, intermediate_steps is always absent.
        self._agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=False,
            max_iterations=10,
            agent_executor_kwargs={"return_intermediate_steps": True, "handle_parsing_errors": True},
        )
        self._build_error = None
        logger.info("[NLP] SQL agent built successfully (db=%s)", diag["database"])

    # ── Security ──────────────────────────────────────────────────────────────

    def _check_prompt_injection(self, question: str) -> bool:
        """Returns True if the question looks like a prompt-injection attempt."""
        return bool(INJECTION_PATTERNS.search(question))

    def _validate_sql(self, sql: str) -> bool:
        """Returns True only if the SQL is safe to expose to the client."""
        if BLOCKED_KEYWORDS.search(sql):
            return False
        sql_upper = sql.upper()
        is_aggregation = bool(re.search(
            r"\b(COUNT|AVG|SUM|MIN|MAX)\s*\(|\bGROUP\s+BY\b", sql_upper
        ))
        if not is_aggregation and "LIMIT" not in sql_upper:
            return False
        sql_for_tables = sql
        while re.search(r'\([^()]*\)', sql_for_tables):
            sql_for_tables = re.sub(r'\([^()]*\)', '', sql_for_tables)
        for table in re.findall(r"\bFROM\s+(\w+)", sql_for_tables, re.IGNORECASE):
            if table.lower() not in ALLOWED_TABLES:
                return False
        return True

    # ── Query ─────────────────────────────────────────────────────────────────

    async def query(self, question: str) -> Dict[str, Any]:
        # Security gate: reject prompt injection before touching the LLM
        if self._check_prompt_injection(question):
            logger.warning("[NLP] Prompt injection attempt blocked: %.80s", question)
            return {
                "answer": "I'm sorry, I can only answer questions about insurance data.",
                "sql_used": None,
                "data": None,
                "chart_suggestion": "none",
            }

        try:
            if self._agent is None:
                self._build_agent()

            # Retry loop — wraps only the LLM call.
            # SQL extraction, chart logic, and formatting run after this block unchanged.
            for _attempt, _wait in enumerate(_RETRY_DELAYS):
                if _wait > 0:
                    await asyncio.sleep(_wait)
                try:
                    result = self._agent.invoke({"input": question})
                    break  # success — exit retry loop
                except Exception as _exc:
                    _is_last = _attempt == len(_RETRY_DELAYS) - 1
                    if not _is_last and _is_retriable(_exc):
                        logger.warning(
                            "[NLP] attempt %d/%d failed (%s), retrying in %ds",
                            _attempt + 1, len(_RETRY_DELAYS),
                            type(_exc).__name__, _RETRY_DELAYS[_attempt + 1],
                        )
                        continue
                    raise  # non-retriable or all attempts exhausted

            raw_answer: str = result.get("output", "")

            # ── Extract SQL + raw observation from intermediate steps ──────
            # intermediate_steps is a list of (AgentAction, observation_str) tuples.
            # We take the LAST sql_db_query action so we capture the final,
            # refined query if the agent retried.
            sql_used: Optional[str] = None
            raw_obs: Optional[str] = None
            for action, observation in result.get("intermediate_steps", []):
                if not hasattr(action, "tool"):
                    continue
                if action.tool == "sql_db_query":
                    candidate = (
                        action.tool_input
                        if isinstance(action.tool_input, str)
                        else str(action.tool_input)
                    )
                    if self._validate_sql(candidate):
                        sql_used = candidate
                        raw_obs = str(observation)
                    else:
                        logger.warning("[NLP] Blocked unsafe SQL from agent: %.200s", candidate)
                        sql_used = "[blocked]"
                        raw_obs = None

            # ── Parse structured data from the SQL result string ──────────
            data: Optional[List[dict]] = None
            if sql_used and sql_used != "[blocked]" and raw_obs:
                data = _parse_sql_observation(raw_obs, sql_used)

            # ── Extract [CHART: xxx] marker the LLM appended ──────────────
            llm_chart_hint: Optional[str] = None
            chart_match = re.search(r"\[CHART:\s*(bar|line|pie|none)\]", raw_answer, re.IGNORECASE)
            if chart_match:
                llm_chart_hint = chart_match.group(1).lower()
                # Strip the marker from the visible answer
                raw_answer = re.sub(
                    r"\s*\[CHART:\s*(bar|line|pie|none)\]\s*$", "", raw_answer, flags=re.IGNORECASE
                ).strip()

            chart_suggestion = _suggest_chart_from_result(data, sql_used or "", llm_chart_hint)

            # Post-process: promote bare numeric LLM answers to full sentences
            raw_answer = _format_answer(raw_answer, data, sql_used or "")

            return {
                "answer": raw_answer,
                "sql_used": sql_used,
                "data": data,
                "chart_suggestion": chart_suggestion,
            }

        except Exception as exc:
            if self._agent is None:
                self._build_error = str(exc)
            logger.error("[NLP] query failed: %s", exc, exc_info=True)
            category = _classify_error(exc)
            if category == "rate_limit":
                safe_answer = (
                    "The AI assistant is temporarily unavailable due to API rate limits. "
                    "Please try again in a few minutes."
                )
            elif category == "sql":
                safe_answer = "The requested information could not be retrieved at this time."
            else:
                safe_answer = "The AI assistant encountered an unexpected error. Please try again."
            return {
                "answer": safe_answer,
                "sql_used": None,
                "data": None,
                "chart_suggestion": "none",
            }


# ── Singleton ─────────────────────────────────────────────────────────────────

_assistant: Optional[InsuranceAssistant] = None


def get_assistant() -> InsuranceAssistant:
    global _assistant
    if _assistant is None:
        _assistant = InsuranceAssistant()
    return _assistant
