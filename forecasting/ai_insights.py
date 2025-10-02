# forecasting/ai_insights.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from django.utils import timezone
import logging
from .models import ForecastResult, ForecastingModel, ExternalFactors

logger = logging.getLogger(__name__)

class AIIntelligenceEngine:
    def __init__(self):
        self.insight_templates = {
            'price_trend': [
                "📈 **Strong Bullish Trend**: Prices are expected to increase by {change_pct}% over the next {days} days. Consider holding inventory for better returns.",
                "📊 **Stable Market**: Prices show minimal fluctuation ({change_pct}% change). Maintain regular sales strategy.",
                "⚠️ **Bearish Pressure**: Prices may decline by {abs_change_pct}% in the coming days. Consider accelerating sales.",
                "🚀 **Rapid Growth**: Exceptional price growth of {change_pct}% detected! Optimal selling window opening."
            ],
            'demand_pattern': [
                "📦 **High Demand Period**: Demand expected to increase by {demand_change_pct}%. Plan for increased production/supply.",
                "📉 **Demand Slowdown**: Lower demand anticipated ({demand_change_pct}% decrease). Focus on quality over quantity.",
                "🔄 **Stable Consumption**: Consistent demand patterns observed. Maintain current operational levels.",
                "🎯 **Peak Demand**: Significant demand surge detected - maximize availability during this period."
            ],
            'revenue_opportunity': [
                "💰 **Revenue Peak**: Potential revenue increase of {revenue_change_pct}% identified. Optimize sales strategy.",
                "💡 **Strategic Timing**: Best revenue day: {best_day} with estimated R{best_revenue:,.0f} potential.",
                "📅 **Weekly Pattern**: {best_day_of_week} typically shows highest revenue performance.",
                "🎪 **Seasonal Opportunity**: Current period shows {seasonal_trend} revenue patterns."
            ],
            'risk_assessment': [
                "🛡️ **Low Risk**: Market conditions appear stable with minimal volatility.",
                "⚠️ **Moderate Risk**: Some price volatility detected. Monitor market closely.",
                "🚨 **High Risk**: Significant market fluctuations expected. Implement risk mitigation strategies.",
                "🔴 **Critical Risk**: Extreme volatility detected. Consider conservative approach."
            ]
        }
    
    def generate_insights(self, model_id):
        """Generate AI-powered insights for a specific forecasting model"""
        try:
            model = ForecastingModel.objects.get(id=model_id)
            forecasts_7 = ForecastResult.objects.filter(model=model, forecast_horizon=7).order_by('forecast_date')
            forecasts_30 = ForecastResult.objects.filter(model=model, forecast_horizon=30).order_by('forecast_date')
            
            if not forecasts_7.exists() or not forecasts_30.exists():
                return self._get_default_insights()
            
            insights = []
            
            # 1. Price Trend Analysis
            price_insights = self._analyze_price_trends(forecasts_7, forecasts_30)
            insights.extend(price_insights)
            
            # 2. Demand Pattern Analysis
            demand_insights = self._analyze_demand_patterns(forecasts_7, forecasts_30)
            insights.extend(demand_insights)
            
            # 3. Revenue Optimization
            revenue_insights = self._analyze_revenue_opportunities(forecasts_7, forecasts_30)
            insights.extend(revenue_insights)
            
            # 4. Risk Assessment
            risk_insights = self._assess_market_risk(forecasts_7, forecasts_30)
            insights.extend(risk_insights)
            
            # 5. Actionable Recommendations
            action_insights = self._generate_actionable_recommendations(forecasts_7, forecasts_30, model)
            insights.extend(action_insights)
            
            # 6. External Factors Impact
            external_insights = self._analyze_external_factors(model)
            insights.extend(external_insights)
            
            return "\n\n".join(insights)
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return self._get_default_insights()
    
    def _analyze_price_trends(self, forecasts_7, forecasts_30):
        """Analyze price trends and generate insights"""
        insights = []
        
        # Convert to lists for analysis
        prices_7 = [float(f.predicted_price) for f in forecasts_7]
        prices_30 = [float(f.predicted_price) for f in forecasts_30]
        
        if len(prices_7) < 2 or len(prices_30) < 2:
            return ["📊 **Price Data**: Insufficient data for detailed trend analysis."]
        
        # Calculate trends
        week_trend = ((prices_7[-1] - prices_7[0]) / prices_7[0]) * 100
        month_trend = ((prices_30[-1] - prices_30[0]) / prices_30[0]) * 100
        
        # Weekly volatility
        week_volatility = (max(prices_7) - min(prices_7)) / np.mean(prices_7) * 100
        
        # Generate insights based on trends
        if week_trend > 5:
            insights.append(f"📈 **Short-term Bullish**: Prices expected to rise {week_trend:.1f}% this week. Consider strategic holding.")
        elif week_trend < -3:
            insights.append(f"⚠️ **Short-term Caution**: Prices may decline {abs(week_trend):.1f}% this week. Accelerate sales if possible.")
        else:
            insights.append("📊 **Stable Short-term**: Minimal price movement expected this week.")
        
        if month_trend > 10:
            insights.append(f"🚀 **Strong Monthly Growth**: {month_trend:.1f}% price increase expected over 30 days. Excellent market conditions!")
        elif month_trend < -5:
            insights.append(f"🔻 **Monthly Downtrend**: {abs(month_trend):.1f}% decline anticipated. Review pricing strategy.")
        
        if week_volatility > 15:
            insights.append(f"🎢 **High Volatility**: {week_volatility:.1f}% price swings expected. Monitor daily for optimal timing.")
        
        return insights
    
    def _analyze_demand_patterns(self, forecasts_7, forecasts_30):
        """Analyze demand patterns and generate insights"""
        insights = []
        
        demands_7 = [float(f.predicted_demand) for f in forecasts_7]
        demands_30 = [float(f.predicted_demand) for f in forecasts_30]
        
        if len(demands_7) < 2:
            return ["📦 **Demand Data**: Analyzing consumption patterns..."]
        
        # Calculate demand metrics
        avg_demand_7 = np.mean(demands_7)
        avg_demand_30 = np.mean(demands_30)
        demand_growth = ((avg_demand_30 - avg_demand_7) / avg_demand_7) * 100 if avg_demand_7 else 0
        
        # Peak demand analysis
        peak_demand_day = forecasts_7.order_by('-predicted_demand').first()
        
        if demand_growth > 20:
            insights.append(f"📦 **Growing Demand**: Consumption expected to increase {demand_growth:.1f}% monthly. Scale operations accordingly.")
        elif demand_growth < -10:
            insights.append(f"📉 **Demand Contraction**: {abs(demand_growth):.1f}% decrease anticipated. Focus on market retention.")
        
        if peak_demand_day:
            insights.append(f"🎯 **Peak Consumption**: Highest demand expected on {peak_demand_day.forecast_date.strftime('%A')} - plan inventory accordingly.")
        
        # Weekly pattern detection
        weekday_demands = {}
        for forecast in forecasts_7:
            weekday = forecast.forecast_date.strftime('%A')
            if weekday not in weekday_demands:
                weekday_demands[weekday] = []
            weekday_demands[weekday].append(float(forecast.predicted_demand))
        
        if weekday_demands:
            best_weekday = max(weekday_demands.keys(), key=lambda x: np.mean(weekday_demands[x]))
            insights.append(f"📅 **Weekly Pattern**: {best_weekday}s typically show strongest demand.")
        
        return insights
    
    def _analyze_revenue_opportunities(self, forecasts_7, forecasts_30):
        """Identify revenue optimization opportunities"""
        insights = []
        
        # Calculate revenue for each day
        daily_revenues = []
        for forecast in forecasts_7:
            revenue = float(forecast.predicted_price) * float(forecast.predicted_demand)
            daily_revenues.append({
                'date': forecast.forecast_date,
                'revenue': revenue,
                'price': float(forecast.predicted_price),
                'demand': float(forecast.predicted_demand)
            })
        
        if not daily_revenues:
            return ["💰 **Revenue Analysis**: Calculating optimal sales timing..."]
        
        # Find best revenue day
        best_day = max(daily_revenues, key=lambda x: x['revenue'])
        total_week_revenue = sum(item['revenue'] for item in daily_revenues)
        avg_daily_revenue = total_week_revenue / len(daily_revenues)
        
        insights.append(f"💎 **Revenue Peak**: {best_day['date'].strftime('%A')} offers best revenue potential: R{best_day['revenue']:,.0f}")
        
        # Price-demand correlation insight
        price_demand_corr = np.corrcoef(
            [item['price'] for item in daily_revenues],
            [item['demand'] for item in daily_revenues]
        )[0,1]
        
        if price_demand_corr > 0.3:
            insights.append("⚡ **Premium Market**: Higher prices correlate with increased demand - premium positioning effective.")
        elif price_demand_corr < -0.3:
            insights.append("💡 **Price Sensitivity**: Demand decreases with price increases - consider competitive pricing.")
        
        return insights
    
    def _assess_market_risk(self, forecasts_7, forecasts_30):
        """Assess market risks and volatility"""
        insights = []
        
        prices_7 = [float(f.predicted_price) for f in forecasts_7]
        prices_30 = [float(f.predicted_price) for f in forecasts_30]
        
        if len(prices_7) < 2:
            return ["🛡️ **Risk Assessment**: Evaluating market stability..."]
        
        # Calculate risk metrics
        price_range_7 = (max(prices_7) - min(prices_7)) / np.mean(prices_7) * 100
        price_std_7 = np.std(prices_7) / np.mean(prices_7) * 100
        
        if price_range_7 > 20:
            insights.append("🚨 **High Volatility**: Significant price swings expected. Implement risk management strategies.")
        elif price_range_7 > 10:
            insights.append("⚠️ **Moderate Fluctuations**: Monitor market closely for optimal entry/exit points.")
        else:
            insights.append("🛡️ **Market Stability**: Low volatility expected - predictable trading conditions.")
        
        # Confidence interval analysis
        confidence_ranges = []
        for forecast in forecasts_7:
            range_width = (float(forecast.price_upper_bound) - float(forecast.price_lower_bound)) / float(forecast.predicted_price) * 100
            confidence_ranges.append(range_width)
        
        avg_confidence_range = np.mean(confidence_ranges) if confidence_ranges else 0
        
        if avg_confidence_range > 15:
            insights.append("🔍 **High Uncertainty**: Wide confidence intervals suggest market unpredictability.")
        
        return insights
    
    def _generate_actionable_recommendations(self, forecasts_7, forecasts_30, model):
        """Generate specific, actionable recommendations"""
        recommendations = []
        
        # Inventory recommendations
        total_7_day_demand = sum(float(f.predicted_demand) for f in forecasts_7)
        avg_daily_demand_7 = total_7_day_demand / 7 if forecasts_7.count() > 0 else 0
        
        recommendations.append(f"📊 **Inventory Planning**: Maintain {avg_daily_demand_7:.0f} kg daily inventory for optimal supply.")
        
        # Pricing strategy
        prices_7 = [float(f.predicted_price) for f in forecasts_7]
        if prices_7:
            price_variance = (max(prices_7) - min(prices_7)) / np.mean(prices_7) * 100
            if price_variance > 15:
                recommendations.append("💰 **Dynamic Pricing**: Significant daily price variations detected - implement flexible pricing strategy.")
            else:
                recommendations.append("🏷️ **Stable Pricing**: Minimal price fluctuations - maintain consistent pricing structure.")
        
        # Seasonal considerations
        current_month = datetime.now().month
        if current_month in [11, 12, 1]:  # Summer months in SA
            recommendations.append("☀️ **Seasonal Factor**: Summer season may affect consumption patterns - monitor closely.")
        elif current_month in [6, 7, 8]:  # Winter months
            recommendations.append("❄️ **Winter Strategy**: Colder months may influence demand - adjust expectations accordingly.")
        
        # Model accuracy consideration
        if model.price_model_accuracy and model.price_model_accuracy < 0.7:
            recommendations.append("🔧 **Model Refinement**: Consider updating training data for improved forecast accuracy.")
        
        return ["**🎯 Recommended Actions:**"] + [f"• {rec}" for rec in recommendations]
    
    def _analyze_external_factors(self, model):
        """Analyze impact of external factors"""
        insights = []
        
        try:
            # Get recent external factors
            recent_factors = ExternalFactors.objects.filter(
                province=model.province
            ).order_by('-date')[:7]
            
            if recent_factors.exists():
                # Analyze weather impact
                recent_temps = [f.avg_temperature for f in recent_factors if f.avg_temperature]
                recent_rainfall = [f.rainfall_mm for f in recent_factors if f.rainfall_mm]
                
                if recent_temps:
                    avg_temp = np.mean(recent_temps)
                    if avg_temp > 25:
                        insights.append("🌡️ **Weather Impact**: High temperatures may affect product quality - ensure proper storage.")
                    elif avg_temp < 10:
                        insights.append("🥶 **Cold Weather**: Lower temperatures might influence consumer behavior.")
                
                if recent_rainfall and any(r > 50 for r in recent_rainfall):
                    insights.append("🌧️ **Rainfall Alert**: Heavy rainfall may impact logistics and supply chain.")
            
            # Holiday impact
            upcoming_holidays = ExternalFactors.objects.filter(
                province=model.province,
                is_holiday=True,
                date__gte=timezone.now().date()
            )[:3]
            
            if upcoming_holidays.exists():
                holiday_dates = [h.date.strftime('%b %d') for h in upcoming_holidays]
                insights.append(f"🎄 **Holiday Season**: Upcoming holidays ({', '.join(holiday_dates)}) may affect market dynamics.")
            
        except Exception as e:
            logger.warning(f"Could not analyze external factors: {e}")
        
        return insights
    
    def _get_default_insights(self):
        """Return default insights when data is insufficient"""
        return [
            "🤖 **AI Market Intelligence**",
            "📊 **Analysis**: Processing market data...",
            "💡 **Initial Insight**: Market conditions are being evaluated for optimal recommendations.",
            "🔄 **Next Steps**: Check back shortly for detailed analysis and actionable insights.",
            "",
            "**📈 Quick Tips:**",
            "• Monitor daily price fluctuations",
            "• Track demand patterns weekly", 
            "• Consider seasonal market trends",
            "• Review external factors regularly"
        ]