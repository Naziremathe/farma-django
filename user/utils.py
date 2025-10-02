from django.core.cache import cache
from django.db import connection
from functools import wraps
import time

def cache_user_data(user_id, user_data, timeout=3600):
    """Cache user data with timeout"""
    cache_key = f'user_{user_id}'
    cache.set(cache_key, user_data, timeout)

def get_cached_user_data(user_id):
    """Get cached user data"""
    cache_key = f'user_{user_id}'
    return cache.get(cache_key)

def query_debugger(func):
    """Decorator to debug database queries"""
    @wraps(func)
    def inner_func(*args, **kwargs):
        start_queries = len(connection.queries)
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_queries = len(connection.queries)
        
        print(f"Function: {func.__name__}")
        print(f"Number of Queries: {end_queries - start_queries}")
        print(f"Finished in: {(end_time - start_time):.2f}s")
        
        return result
    return inner_func