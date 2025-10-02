import os
import openai

# Set your API key (or use environment variable)
OPENAI_API_KEY= ''

def gpt_insights(forecast_price, forecast_demand, crop_name="Potatoes", province="KZN"):
    # Prepare a simple prompt
    prompt = f"""
    I have forecasted {crop_name} prices in {province} for the next {len(forecast_price)} days:
    {forecast_price[['ds', 'yhat']].to_dict(orient='records')}
    
    Forecasted demand (kg):
    {forecast_demand[['ds', 'yhat']].to_dict(orient='records')}
    
    Provide insights in **simple language for farmers**:
    - Best days to sell
    - Expected price range
    - Any risks (like low stock, heavy rain, fuel price changes)
    """
    
    # Call OpenAI GPT API
    response = openai.Completion.create(
        model="text-davinci-003",  # or "gpt-3.5-turbo"
        prompt=prompt,
        temperature=0.7,
        max_tokens=250
    )
    
    return response.choices[0].text.strip()
