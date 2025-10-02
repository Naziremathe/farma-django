# Create this file: your_app/templatetags/forecast_filters.py

from django import template
from django.utils.safestring import mark_safe
import locale

register = template.Library()

@register.filter
def mul(value, arg):
    """Multiply the arg by the value."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter  
def currency(value, decimal_places=0):
    """Format number as South African currency"""
    try:
        if decimal_places == 0:
            return f"R{float(value):,.0f}"
        else:
            return f"R{float(value):,.{decimal_places}f}"
    except (ValueError, TypeError):
        return "R0"

@register.filter
def percentage(value, decimal_places=1):
    """Format number as percentage"""
    try:
        return f"{float(value):.{decimal_places}f}%"
    except (ValueError, TypeError):
        return "0.0%"

@register.filter
def kg_format(value):
    """Format weight in kilograms"""
    try:
        return f"{float(value):,.0f} kg"
    except (ValueError, TypeError):
        return "0 kg"

@register.filter
def trend_icon(current, previous):
    """Return trend icon based on comparison"""
    try:
        current_val = float(current)
        previous_val = float(previous)
        
        if current_val > previous_val:
            return mark_safe('<i class="fas fa-arrow-up text-green-600"></i>')
        elif current_val < previous_val:
            return mark_safe('<i class="fas fa-arrow-down text-red-600"></i>')
        else:
            return mark_safe('<i class="fas fa-minus text-gray-600"></i>')
    except (ValueError, TypeError):
        return mark_safe('<i class="fas fa-circle text-blue-600"></i>')

@register.filter
def confidence_badge(confidence_range, predicted_value):
    """Return confidence level badge"""
    try:
        range_val = float(confidence_range)
        predicted_val = float(predicted_value)
        
        if predicted_val == 0:
            return mark_safe('<span class="badge bg-secondary">Unknown</span>')
        
        confidence_pct = (range_val / predicted_val) * 100
        
        if confidence_pct < 20:
            return mark_safe('<span class="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-semibold">High Confidence</span>')
        elif confidence_pct < 40:
            return mark_safe('<span class="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs font-semibold">Medium Confidence</span>')
        else:
            return mark_safe('<span class="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs font-semibold">Low Confidence</span>')
    except (ValueError, TypeError):
        return mark_safe('<span class="px-2 py-1 bg-gray-100 text-gray-800 rounded-full text-xs font-semibold">Unknown</span>')

@register.filter
def market_sentiment_color(sentiment):
    """Return CSS classes for market sentiment"""
    sentiment_lower = str(sentiment).lower()
    
    if sentiment_lower == 'bullish':
        return 'text-green-600 bg-green-100'
    elif sentiment_lower == 'bearish':
        return 'text-red-600 bg-red-100'
    else:
        return 'text-yellow-600 bg-yellow-100'

@register.filter
def risk_level_color(risk_level):
    """Return CSS classes for risk level"""
    risk_lower = str(risk_level).lower()
    
    if risk_lower == 'low':
        return 'text-green-600'
    elif risk_lower == 'medium':
        return 'text-yellow-600'
    else:
        return 'text-red-600'

@register.simple_tag
def revenue_calculation(quantity, price):
    """Calculate revenue from quantity and price"""
    try:
        return float(quantity) * float(price)
    except (ValueError, TypeError):
        return 0

@register.inclusion_tag('partials/forecast_card.html')
def forecast_card(title, value, subtitle, icon, color_class="blue"):
    """Render a forecast card component"""
    return {
        'title': title,
        'value': value,
        'subtitle': subtitle,
        'icon': icon,
        'color_class': color_class
    }

@register.inclusion_tag('partials/trend_indicator.html')
def trend_indicator(current_value, change_percent, label):
    """Render trend indicator with arrow and percentage"""
    return {
        'current_value': current_value,
        'change_percent': change_percent,
        'label': label,
        'is_positive': float(change_percent) > 0 if change_percent else False
    }

@register.filter
def get_item(dictionary, key):
    """Get item from dictionary by key"""
    return dictionary.get(key)

@register.filter
def slice_forecast(forecast_list, index):
    """Get specific item from forecast list"""
    try:
        return forecast_list[int(index)]
    except (IndexError, ValueError, TypeError):
        return None
    

@register.filter
def div(value, arg):
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError):
        return 0

@register.filter(name='abs_value')
def abs_value(value):
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return 0



