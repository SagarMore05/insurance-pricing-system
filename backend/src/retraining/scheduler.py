import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import settings

logger = logging.getLogger("scheduler")

_scheduler: BackgroundScheduler = None


def _run_retraining_job():
    from src.retraining.pipeline import RetrainingPipeline

    engine = create_engine(settings.DATABASE_URL_SYNC)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        pipeline = RetrainingPipeline(db_session=session)
        report = pipeline.run()
        logger.info("Scheduled retraining complete: %s", report)
    except Exception as e:
        logger.error("Scheduled retraining failed: %s", e)
    finally:
        session.close()
        engine.dispose()


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _run_retraining_job,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="weekly_retraining",
        name="Weekly model retraining",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.start()
    logger.info("APScheduler started — weekly retraining job registered (Sunday 02:00 UTC)")
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
