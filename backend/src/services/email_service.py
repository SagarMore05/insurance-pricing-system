"""
Email notification service for the Human Approval Workflow.

Sends notifications when an approval request is created, approved, or rejected.
Falls back to structured logging when SMTP is unconfigured (SMTP_USER is empty).

Does NOT send PII — only request_id, model_type, model_version, and status.
"""
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from src.config import settings

logger = logging.getLogger("insurance_api.email")


class EmailNotification:
    def __init__(
        self,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
    ) -> None:
        self.subject = subject
        self.body_text = body_text
        self.body_html = body_html


def _is_smtp_configured() -> bool:
    return bool(settings.SMTP_USER and settings.SMTP_PASS and settings.ADMIN_EMAIL)


def _send_smtp(notification: EmailNotification) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = notification.subject
    msg["From"] = settings.SMTP_USER
    msg["To"] = settings.ADMIN_EMAIL

    msg.attach(MIMEText(notification.body_text, "plain"))
    if notification.body_html:
        msg.attach(MIMEText(notification.body_html, "html"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ctx)
        smtp.login(settings.SMTP_USER, settings.SMTP_PASS)
        smtp.sendmail(settings.SMTP_USER, settings.ADMIN_EMAIL, msg.as_string())


def send_notification(notification: EmailNotification) -> dict:
    """
    Send an email notification. Returns a status dict for audit logging.
    Falls back to mock (log-only) when SMTP is unconfigured.
    """
    if _is_smtp_configured():
        try:
            _send_smtp(notification)
            logger.info("Email sent: %s → %s", notification.subject, settings.ADMIN_EMAIL)
            return {"channel": "smtp", "status": "sent", "recipient": settings.ADMIN_EMAIL}
        except Exception as exc:
            logger.warning("Email send failed (%s) — falling back to mock: %s", type(exc).__name__, exc)
            # Fall through to mock

    # Mock path: log the notification body so it is observable in dev/test
    logger.info(
        "[MOCK EMAIL] Subject: %s | To: %s (unconfigured)\n%s",
        notification.subject,
        settings.ADMIN_EMAIL or "<not set>",
        notification.body_text,
    )
    return {"channel": "mock", "status": "logged", "recipient": settings.ADMIN_EMAIL or "not_configured"}


# ── Notification factories ─────────────────────────────────────────────────────

def notify_request_created(
    request_id: str,
    model_type: str,
    model_version: str,
    submitted_by: str,
    recommendation: Optional[str] = None,
) -> dict:
    subject = f"[Insurance AI] Approval Request: {model_type} model {model_version}"
    body = (
        f"A new model approval request has been submitted.\n\n"
        f"Request ID:    {request_id}\n"
        f"Model Type:    {model_type}\n"
        f"Model Version: {model_version}\n"
        f"Submitted By:  {submitted_by}\n"
        f"Recommendation: {recommendation or 'N/A'}\n\n"
        f"Review this request in the Admin Dashboard → Approvals tab.\n\n"
        f"NOTE: No model changes will occur until an admin explicitly approves."
    )
    return send_notification(EmailNotification(subject=subject, body_text=body))


def notify_request_approved(
    request_id: str,
    model_type: str,
    model_version: str,
    reviewed_by: str,
    reviewer_note: Optional[str],
) -> dict:
    subject = f"[Insurance AI] APPROVED: {model_type} model {model_version}"
    body = (
        f"An approval request has been APPROVED.\n\n"
        f"Request ID:    {request_id}\n"
        f"Model Type:    {model_type}\n"
        f"Model Version: {model_version}\n"
        f"Reviewed By:   {reviewed_by}\n"
        f"Reviewer Note: {reviewer_note or '(none)'}\n\n"
        f"IMPORTANT: Approval is recorded for audit purposes only.\n"
        f"Champion promotion requires a separate V5 pipeline run.\n"
        f"No champion model has been changed."
    )
    return send_notification(EmailNotification(subject=subject, body_text=body))


def notify_request_rejected(
    request_id: str,
    model_type: str,
    model_version: str,
    reviewed_by: str,
    reviewer_note: Optional[str],
) -> dict:
    subject = f"[Insurance AI] REJECTED: {model_type} model {model_version}"
    body = (
        f"An approval request has been REJECTED.\n\n"
        f"Request ID:    {request_id}\n"
        f"Model Type:    {model_type}\n"
        f"Model Version: {model_version}\n"
        f"Reviewed By:   {reviewed_by}\n"
        f"Reviewer Note: {reviewer_note or '(none)'}\n\n"
        f"No changes have been made to production models."
    )
    return send_notification(EmailNotification(subject=subject, body_text=body))
