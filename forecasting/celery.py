# fama/celery.py
import os
from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fama.settings')

app = Celery('fama')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

# Periodic task schedule
app.conf.beat_schedule = {
    'refresh-forecasts-weekly': {
        'task': 'forecasting.tasks.refresh_all_forecasts',
        'schedule': crontab(hour=2, minute=0, day_of_week=1),  # Every Monday at 2 AM
    },
}

app.conf.task_routes = {
    'forecasting.tasks.initialize_forecast_model': {'queue': 'training'},
    'forecasting.tasks.generate_forecasts': {'queue': 'forecasts'},
}

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')