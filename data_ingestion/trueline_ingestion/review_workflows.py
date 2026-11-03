"""Persistent human approval and collaboration records for review workflows."""
from __future__ import annotations

import uuid
from psycopg.types.json import Jsonb


def seed_extracted_rules(conn, brand_id: str, source_id: str, version_id: str, rules: list[dict]) -> None:
    for rule in rules:
        review_id = f"rr_{uuid.uuid4().hex}"
        row = conn.execute(
            "INSERT INTO extracted_rule_reviews (review_id, brand_id, source_id, source_version_id, rule_id, rule_data, updated_by) "
            "VALUES (%s,%s,%s,%s,%s,%s,'system') ON CONFLICT (source_version_id,rule_id) DO NOTHING RETURNING review_id,rule_data,revision",
            (review_id, brand_id, source_id, version_id, rule["rule_id"], Jsonb(rule)),
        ).fetchone()
        if row:
            conn.execute(
                "INSERT INTO extracted_rule_review_audit (review_id,revision,action,actor,next_rule) VALUES (%s,%s,'created','system',%s)",
                (row["review_id"], row["revision"], Jsonb(row["rule_data"])),
            )


def list_extracted_rules(conn, brand_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT r.review_id,r.source_id,r.source_version_id,r.rule_id,r.rule_data,r.status,r.revision,r.updated_by,r.created_at,r.updated_at,s.title "
        "FROM extracted_rule_reviews r JOIN knowledge_sources s ON s.source_id=r.source_id AND s.brand_id=r.brand_id "
        "JOIN knowledge_source_versions v ON v.version_id=r.source_version_id AND v.source_id=s.source_id "
        "WHERE r.brand_id=%s AND v.status='active' AND s.current_version_id=v.version_id ORDER BY r.created_at DESC,r.rule_id", (brand_id,),
    ).fetchall()
    return [{**dict(row), "rule": row["rule_data"], "created_at": row["created_at"].isoformat(),
             "updated_at": row["updated_at"].isoformat()} for row in rows]


def update_extracted_rule(conn, brand_id: str, review_id: str, expected_revision: int,
                          rule: dict, status: str, actor: str) -> dict:
    if status not in {"pending", "approved", "rejected"}:
        raise ValueError("Rule status must be pending, approved, or rejected.")
    row = conn.execute(
        "SELECT r.rule_data,r.revision FROM extracted_rule_reviews r JOIN knowledge_sources s ON s.source_id=r.source_id AND s.brand_id=r.brand_id "
        "JOIN knowledge_source_versions v ON v.version_id=r.source_version_id AND v.source_id=s.source_id "
        "WHERE r.review_id=%s AND r.brand_id=%s AND v.status='active' AND s.current_version_id=v.version_id FOR UPDATE OF r",
        (review_id, brand_id),
    ).fetchone()
    if row is None:
        raise LookupError("Extracted rule was not found.")
    if int(row["revision"]) != int(expected_revision):
        raise RuntimeError("This rule changed since it was opened. Refresh the rules and try again.")
    prior = row["rule_data"]
    if (rule.get("rule_id") != prior.get("rule_id")
            or rule.get("kind") != prior.get("kind")
            or rule.get("citations") != prior.get("citations")
            or not isinstance(rule.get("applies_to_channels", []), list)):
        raise ValueError("Keep the rule identity, type, and exact source citations unchanged; edit only its statement and channel scope.")
    if not isinstance(rule.get("statement"), str) or not rule["statement"].strip() or len(rule["statement"]) > 2000:
        raise ValueError("Rule statement must contain 1–2,000 characters.")
    revision = int(row["revision"]) + 1
    action = "approved" if status == "approved" else "rejected" if status == "rejected" else "edited"
    conn.execute(
        "UPDATE extracted_rule_reviews SET rule_data=%s,status=%s,revision=%s,updated_by=%s,updated_at=now() WHERE review_id=%s",
        (Jsonb(rule), status, revision, actor, review_id),
    )
    conn.execute(
        "INSERT INTO extracted_rule_review_audit (review_id,revision,action,actor,prior_rule,next_rule) VALUES (%s,%s,%s,%s,%s,%s)",
        (review_id, revision, action, actor, Jsonb(prior), Jsonb(rule)),
    )
    conn.commit()
    return {"review_id": review_id, "rule": rule, "status": status, "revision": revision, "updated_by": actor}


def list_rule_audit(conn, brand_id: str, review_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT a.revision,a.action,a.actor,a.prior_rule,a.next_rule,a.created_at FROM extracted_rule_review_audit a "
        "JOIN extracted_rule_reviews r ON r.review_id=a.review_id WHERE r.brand_id=%s AND a.review_id=%s ORDER BY a.revision DESC",
        (brand_id, review_id),
    ).fetchall()
    return [{**dict(row), "created_at": row["created_at"].isoformat()} for row in rows]


def create_review_case(conn, review_id: str, session_id: str, brand_id: str, document_name: str, findings: list) -> None:
    conn.execute(
        "INSERT INTO review_cases (review_id,session_id,brand_id,document_name,findings) VALUES (%s,%s,%s,%s,%s)",
        (review_id, session_id, brand_id, document_name[:255], Jsonb(findings)),
    )
    for index in range(len(findings)):
        conn.execute(
            "INSERT INTO review_finding_decisions (review_id,finding_index,decided_by) VALUES (%s,%s,'system')",
            (review_id, index),
        )
    conn.commit()


def list_review_cases(conn, session_id: str, brand_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT review_id,document_name,findings,assigned_to,status,created_at,updated_at FROM review_cases "
        "WHERE session_id=%s AND brand_id=%s ORDER BY created_at DESC", (session_id, brand_id),
    ).fetchall()
    result = []
    for row in rows:
        item = {**dict(row), "review_id": str(row["review_id"]),
                "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat()}
        item["comments"] = list_comments(conn, row["review_id"])
        item["findings"] = _attach_decisions(conn, row["review_id"], item["findings"])
        result.append(item)
    return result


def get_review_case(conn, review_id: str, session_id: str, brand_id: str) -> dict:
    row = conn.execute(
        "SELECT review_id,document_name,findings,assigned_to,status,created_at,updated_at FROM review_cases "
        "WHERE review_id=%s AND session_id=%s AND brand_id=%s", (review_id, session_id, brand_id),
    ).fetchone()
    if row is None:
        raise LookupError("Review was not found in this session.")
    result = {**dict(row), "review_id": str(row["review_id"]), "comments": list_comments(conn, row["review_id"]),
            "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat()}
    result["findings"] = _attach_decisions(conn, row["review_id"], result["findings"])
    return result


def update_review_case(conn, review_id: str, session_id: str, brand_id: str,
                       assigned_to: str, status: str) -> dict:
    if status not in {"open", "in_review", "approved", "changes_requested"}:
        raise ValueError("Choose a supported review status.")
    cur = conn.execute(
        "UPDATE review_cases SET assigned_to=%s,status=%s,updated_at=now() WHERE review_id=%s AND session_id=%s AND brand_id=%s",
        (assigned_to[:160], status, review_id, session_id, brand_id),
    )
    if cur.rowcount == 0:
        raise LookupError("Review was not found in this session.")
    conn.commit()
    return get_review_case(conn, review_id, session_id, brand_id)


def add_comment(conn, review_id: str, session_id: str, brand_id: str, author: str,
                body: str, finding_index: int | None) -> dict:
    case = get_review_case(conn, review_id, session_id, brand_id)
    if finding_index is not None and not 0 <= finding_index < len(case["findings"]):
        raise ValueError("The selected finding does not exist in this review.")
    row = conn.execute(
        "INSERT INTO review_comments (review_id,author,body,finding_index) VALUES (%s,%s,%s,%s) RETURNING comment_id,author,body,finding_index,created_at",
        (review_id, author[:160], body[:4000], finding_index),
    ).fetchone()
    conn.commit()
    return {**dict(row), "created_at": row["created_at"].isoformat()}


def list_comments(conn, review_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT comment_id,author,body,finding_index,created_at FROM review_comments WHERE review_id=%s ORDER BY created_at,comment_id",
        (review_id,),
    ).fetchall()
    return [{**dict(row), "created_at": row["created_at"].isoformat()} for row in rows]


def update_finding_decision(conn, review_id: str, session_id: str, brand_id: str,
                            finding_index: int, status: str, actor: str) -> dict:
    case = get_review_case(conn, review_id, session_id, brand_id)
    if not 0 <= finding_index < len(case["findings"]):
        raise ValueError("The selected finding does not exist in this review.")
    if status not in {"open", "approved", "changes_requested"}:
        raise ValueError("Choose open, approved, or changes requested.")
    row = conn.execute(
        "INSERT INTO review_finding_decisions (review_id,finding_index,status,decided_by) VALUES (%s,%s,%s,%s) "
        "ON CONFLICT (review_id,finding_index) DO UPDATE SET status=EXCLUDED.status,decided_by=EXCLUDED.decided_by,revision=review_finding_decisions.revision+1,updated_at=now() "
        "RETURNING finding_index,status,decided_by,revision,updated_at",
        (review_id, finding_index, status, actor[:160]),
    ).fetchone()
    conn.commit()
    return {**dict(row), "updated_at": row["updated_at"].isoformat()}


def _attach_decisions(conn, review_id, findings):
    rows = conn.execute(
        "SELECT finding_index,status,decided_by,revision,updated_at FROM review_finding_decisions WHERE review_id=%s",
        (review_id,),
    ).fetchall()
    decisions = {row["finding_index"]: dict(row) for row in rows}
    result = []
    for index, finding in enumerate(findings or []):
        item = dict(finding)
        decision = decisions.get(index, {"status": "open", "decided_by": "system", "revision": 0})
        item["review_decision"] = {**decision, "updated_at": decision["updated_at"].isoformat()
                                   if decision.get("updated_at") else None}
        result.append(item)
    return result
