"""
Human Approval Workflow API — Phase 4B.

Endpoints for creating, listing, reviewing, and auditing model approval requests.

IMPORTANT constraints:
  - No automatic promotion — approval is governance record only
  - No champion registry modification
  - No model file changes
  - No pricing engine changes
  - Retraining remains disabled (ENABLE_SCHEDULED_RETRAINING = False)

All state-changing endpoints require JWT admin authentication.
List / detail endpoints also require auth for auditability.
"""
import math
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import verify_admin_key
from src.api.schemas import (
    ApprovalAuditLogEntry,
    ApprovalRequestCreate,
    ApprovalRequestDetailResponse,
    ApprovalRequestListResponse,
    ApprovalRequestResponse,
    ApprovalReviewAction,
)
from src.database.models import ApprovalAuditLog, ApprovalRequest
from src.database.session import get_db
from src.services.email_service import (
    notify_request_approved,
    notify_request_created,
    notify_request_rejected,
)
from src.services.model_card import generate_model_card

router = APIRouter(dependencies=[Depends(verify_admin_key)])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_dt(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _row_to_response(row: ApprovalRequest) -> ApprovalRequestResponse:
    return ApprovalRequestResponse(
        request_id=str(row.request_id),
        model_type=row.model_type,
        model_version=row.model_version,
        status=row.status,
        submitted_by=row.submitted_by,
        submitted_at=_fmt_dt(row.submitted_at) or "",
        reviewed_by=row.reviewed_by,
        reviewed_at=_fmt_dt(row.reviewed_at),
        reviewer_note=row.reviewer_note,
        model_card=row.model_card,
        recommendation=row.recommendation,
    )


def _row_to_detail(row: ApprovalRequest) -> ApprovalRequestDetailResponse:
    base = _row_to_response(row)
    logs = [
        ApprovalAuditLogEntry(
            log_id=str(log.log_id),
            request_id=str(log.request_id),
            event_type=log.event_type,
            actor=log.actor,
            note=log.note,
            event_data=log.event_data,
            created_at=_fmt_dt(log.created_at) or "",
        )
        for log in (row.audit_logs or [])
    ]
    return ApprovalRequestDetailResponse(**base.model_dump(), audit_logs=logs)


async def _get_or_404(request_id: str, db: AsyncSession) -> ApprovalRequest:
    try:
        rid = uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    result = await db.execute(
        select(ApprovalRequest)
        .options(selectinload(ApprovalRequest.audit_logs))
        .where(ApprovalRequest.request_id == rid)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval request not found")
    return row


async def _add_audit_log(
    db: AsyncSession,
    request_id: uuid.UUID,
    event_type: str,
    actor: Optional[str] = None,
    note: Optional[str] = None,
    event_data: Optional[dict] = None,
) -> None:
    log = ApprovalAuditLog(
        log_id=uuid.uuid4(),
        request_id=request_id,
        event_type=event_type,
        actor=actor,
        note=note,
        event_data=event_data,
        created_at=datetime.utcnow(),
    )
    db.add(log)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/admin/approvals",
    response_model=ApprovalRequestDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit model approval request",
)
async def create_approval_request(
    body: ApprovalRequestCreate,
    admin: dict = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
) -> ApprovalRequestDetailResponse:
    """
    Submit a new approval request for a model type (frequency or severity).

    Auto-generates the model card from champion_registry.json + latest drift snapshot.
    Does NOT promote a model, does NOT modify champions.
    """
    if body.model_type not in ("frequency", "severity"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="model_type must be 'frequency' or 'severity'",
        )

    card = generate_model_card(body.model_type)
    submitted_by = admin.get("sub", "admin")
    now = datetime.utcnow()

    row = ApprovalRequest(
        request_id=uuid.uuid4(),
        model_type=body.model_type,
        model_version=card["model_version"],
        status="pending",
        submitted_by=submitted_by,
        submitted_at=now,
        model_card=card,
        recommendation=card.get("recommendation"),
    )
    db.add(row)
    await db.flush()  # get request_id populated

    await _add_audit_log(
        db,
        row.request_id,
        event_type="created",
        actor=submitted_by,
        note=body.notes,
        event_data={
            "model_version": card["model_version"],
            "algorithm": card.get("algorithm"),
            "dataset_size": card.get("dataset_size"),
            "recommendation": card.get("recommendation"),
        },
    )

    email_result = notify_request_created(
        request_id=str(row.request_id),
        model_type=body.model_type,
        model_version=card["model_version"],
        submitted_by=submitted_by,
        recommendation=card.get("recommendation"),
    )
    await _add_audit_log(
        db,
        row.request_id,
        event_type="email_notification",
        actor="system",
        note=f"channel={email_result['channel']} status={email_result['status']}",
        event_data=email_result,
    )

    await db.commit()

    # Re-fetch with audit logs eager-loaded
    row = await _get_or_404(str(row.request_id), db)
    return _row_to_detail(row)


@router.get(
    "/admin/approvals",
    response_model=ApprovalRequestListResponse,
    summary="List approval requests",
)
async def list_approval_requests(
    status_filter: Optional[str] = Query(None, alias="status"),
    model_type: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> ApprovalRequestListResponse:
    """
    List approval requests. Supports filtering by status and model_type.
    Results ordered by submitted_at descending (newest first).
    """
    filters = []
    if status_filter:
        filters.append(ApprovalRequest.status == status_filter)
    if model_type:
        filters.append(ApprovalRequest.model_type == model_type)

    count_q = select(func.count()).select_from(ApprovalRequest)
    if filters:
        count_q = count_q.where(and_(*filters))
    total = (await db.execute(count_q)).scalar() or 0

    rows_q = (
        select(ApprovalRequest)
        .order_by(ApprovalRequest.submitted_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    if filters:
        rows_q = rows_q.where(and_(*filters))

    rows = (await db.execute(rows_q)).scalars().all()
    total_pages = max(1, math.ceil(total / page_size))

    return ApprovalRequestListResponse(
        items=[_row_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/admin/approvals/{request_id}",
    response_model=ApprovalRequestDetailResponse,
    summary="Get approval request detail",
)
async def get_approval_request(
    request_id: str,
    db: AsyncSession = Depends(get_db),
) -> ApprovalRequestDetailResponse:
    """Return full detail for an approval request, including audit log."""
    row = await _get_or_404(request_id, db)
    return _row_to_detail(row)


@router.post(
    "/admin/approvals/{request_id}/approve",
    response_model=ApprovalRequestDetailResponse,
    summary="Approve an approval request",
)
async def approve_approval_request(
    request_id: str,
    body: ApprovalReviewAction,
    admin: dict = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
) -> ApprovalRequestDetailResponse:
    """
    Approve an approval request.

    Records the decision, reviewer identity, and note in the audit log.
    Does NOT promote the model or modify the champion registry.
    Sends an email notification (mock if SMTP unconfigured).
    """
    row = await _get_or_404(request_id, db)

    if row.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Request is already '{row.status}' — cannot approve.",
        )

    reviewer = admin.get("sub", "admin")
    now = datetime.utcnow()

    row.status = "approved"
    row.reviewed_by = reviewer
    row.reviewed_at = now
    row.reviewer_note = body.reviewer_note

    await _add_audit_log(
        db,
        row.request_id,
        event_type="approved",
        actor=reviewer,
        note=body.reviewer_note,
        event_data={"previous_status": "pending", "model_version": row.model_version},
    )

    email_result = notify_request_approved(
        request_id=str(row.request_id),
        model_type=row.model_type,
        model_version=row.model_version,
        reviewed_by=reviewer,
        reviewer_note=body.reviewer_note,
    )
    await _add_audit_log(
        db,
        row.request_id,
        event_type="email_notification",
        actor="system",
        note=f"channel={email_result['channel']} status={email_result['status']}",
        event_data=email_result,
    )

    await db.commit()
    row = await _get_or_404(str(row.request_id), db)
    return _row_to_detail(row)


@router.post(
    "/admin/approvals/{request_id}/reject",
    response_model=ApprovalRequestDetailResponse,
    summary="Reject an approval request",
)
async def reject_approval_request(
    request_id: str,
    body: ApprovalReviewAction,
    admin: dict = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
) -> ApprovalRequestDetailResponse:
    """
    Reject an approval request.

    Records the decision, reviewer identity, and note in the audit log.
    Does NOT modify production models.
    """
    row = await _get_or_404(request_id, db)

    if row.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Request is already '{row.status}' — cannot reject.",
        )

    reviewer = admin.get("sub", "admin")
    now = datetime.utcnow()

    row.status = "rejected"
    row.reviewed_by = reviewer
    row.reviewed_at = now
    row.reviewer_note = body.reviewer_note

    await _add_audit_log(
        db,
        row.request_id,
        event_type="rejected",
        actor=reviewer,
        note=body.reviewer_note,
        event_data={"previous_status": "pending", "model_version": row.model_version},
    )

    email_result = notify_request_rejected(
        request_id=str(row.request_id),
        model_type=row.model_type,
        model_version=row.model_version,
        reviewed_by=reviewer,
        reviewer_note=body.reviewer_note,
    )
    await _add_audit_log(
        db,
        row.request_id,
        event_type="email_notification",
        actor="system",
        note=f"channel={email_result['channel']} status={email_result['status']}",
        event_data=email_result,
    )

    await db.commit()
    row = await _get_or_404(str(row.request_id), db)
    return _row_to_detail(row)


@router.get(
    "/admin/approvals/{request_id}/audit-log",
    response_model=List[ApprovalAuditLogEntry],
    summary="Get audit log for an approval request",
)
async def get_approval_audit_log(
    request_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[ApprovalAuditLogEntry]:
    """Return the full immutable audit log for a specific approval request."""
    row = await _get_or_404(request_id, db)
    return [
        ApprovalAuditLogEntry(
            log_id=str(log.log_id),
            request_id=str(log.request_id),
            event_type=log.event_type,
            actor=log.actor,
            note=log.note,
            event_data=log.event_data,
            created_at=_fmt_dt(log.created_at) or "",
        )
        for log in (row.audit_logs or [])
    ]
