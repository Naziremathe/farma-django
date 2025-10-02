# fama/__init__.py
from forecasting.celery import app as celery_app

__all__ = ('celery_app',)