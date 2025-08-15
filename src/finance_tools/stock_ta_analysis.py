from src.main import tool
from src.main import Union,Dict
from src.main import DuckDuckGoSearchRun
from src.main import requests
from src.main import List, TypedDict, Any, Tuple, Union, Dict, Set, Annotated, Optional
from src.main import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from src.main import BaseModel, Field
from src.utils.api_helpers import llm_claude35

@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches current and historical stock price data for a given ticker."""
    try:
        import datetime as dt
        import yfinance as yf

        # Get stock data for the last 3 months
        stock = yf.Ticker(ticker)
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(days=90),
            end=dt.datetime.now(),
            interval='1d'
        )

        if data.empty:
            return f"No data found for ticker {ticker}"

        try:
            current_price = float(data['Close'].iloc[-1])
            previous_close = float(data['Close'].iloc[-2])
            current_volume = float(data['Volume'].iloc[-1])

            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100
            high_90d = float(data['High'].max())
            low_90d = float(data['Low'].min())
            avg_volume = float(data['Volume'].mean())

            return {
                "stock": ticker,
                "current_price": round(current_price, 2),
                "previous_close": round(previous_close, 2),
                "price_change": round(price_change, 2),
                "price_change_percent": round(price_change_percent, 2),
                "volume": int(current_volume),
                "high_90d": round(high_90d, 2),
                "low_90d": round(low_90d, 2),
                "average_volume": int(avg_volume),
                "date": dt.datetime.now().strftime("%Y-%m-%d")
            }

        except IndexError:
            return f"Insufficient data for ticker {ticker}"

    except Exception as e:
        return f"Error fetching price data: {str(e)}"



@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Fetches key financial ratios for a given ticker."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'pe_ratio': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'profit_margins': info.get('profitMargins'),
            'previous_close': stock.info['previousClose']
        }
    except Exception as e:
        return f"Error fetching ratios: {str(e)}"

search = DuckDuckGoSearchRun()


class CompanyFinancials(BaseModel):
    symbol:str =  Field(description="The symbol of the company")
    companyName:str =  Field(description="The name of the company")
    marketCap:float = Field(alias="mktCap", description="The market capitalization of the company")
    industry:str =  Field(description="The industry of the company")
    sector:str =  Field(description="The sector of the company")
    description:str = Field(description="The description of the company")
    website:str =  Field(description="The website of the company")
    beta:float = Field(description="The beta of the company")
    price:float = Field(description="The price of the company")

@tool
def get_company_financials(symbol) -> Tuple[Any, CompanyFinancials]:
    """
    Fetch basic financial information for the given company symbol such as the industry, the sector, the name of the company, and the market capitalization.
    """
    try:
      url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
      response = requests.get(url)
      data = response.json()
      financials = CompanyFinancials(**data[0])
      return financials
    except (IndexError, KeyError):
        return {"error": f"Could not fetch financials for symbol: {symbol}"}

# Update the tools list to include the stock price function
tools = [search, get_stock_prices, get_financial_metrics, get_company_financials]

llm_with_tools = llm_claude35.bind_tools(tools)

FUNDAMENTAL_ANALYST_PROMPT = """
You are a fundamental analyst specializing in evaluating company performance based on stock prices, technical indicators, and financial metrics. Your task is to provide a comprehensive summary of the fundamental analysis for a given company.

You have access to the following tools:
1. **search**: to find information relevant with the question to get company symbols so that you can use tools more effectively
2. **get_stock_prices**: Retrieves the latest stock price, historical price data and technical Indicators like 90 days high and low.
3. **get_financial_metrics**: Retrieves key financial metrics, such as revenue, earnings per share (EPS), price-to-earnings ratio (P/E), and debt-to-equity ratio.
4. **get_company_financials**: Retrieves key company information, such as description of the company, industry, etc.

### Your Task:
1. **Search**: Use the search tool to search relevant information to finish necessary task on stock analysis.
2. **Analyze Data**: Evaluate the results from the tools and identify potential resistance, key trends, strengths, or concerns.
3. **Provide Summary**: Write a concise, well-structured summary that highlights:
    - Recent stock price movements, trends and potential resistance.
    - Key insights from technical indicators (e.g., whether the stock is overbought or oversold).
    - Financial health and performance based on financial metrics.

### Constraints:
- Avoid speculative language; focus on observable data and trends.
- If any tool fails to provide data, clearly state that in your summary.

### Output Format:
Respond in the following format:
"stock": "<Stock Symbol>",
"price_analysis": "<Detailed analysis of stock price trends>",
"technical_analysis": "<Detailed time series Analysis from ALL technical indicators>",
"financial_analysis": "<Detailed analysis from financial metrics>",
"final Summary": "<Full Conclusion based on the above analyses>"
"Asked Question Answer": "<Answer based on the details and analysis above>"

Ensure that your response is objective, concise, and actionable."""


def reasoner(state):
    """
    Fundamental analysis reasoner function
    """
    query = state["question"]
    messages = state["messages"]
    result = []

    # System message indicating the assistant's capabilities
    sys_msg = SystemMessage(content=FUNDAMENTAL_ANALYST_PROMPT)
    message = HumanMessage(content=query)
    messages.append(message)

    # Invoke the LLM with the messages
    result = [llm_with_tools.invoke([sys_msg] + messages)]

    # Print the response steps
    print("\n=== Reasoner Analysis Steps ===")
    for idx, m in enumerate(result, 1):
        print(f"\nStep {idx}:")
        m.pretty_print()

    # When analysis is complete
    if result and "complete analysis" in result[-1].content.lower():
        return {
            "messages": result,
            "__end__": True  # Signal to end the conversation
        }

    # If analysis is not complete
    return {
        "messages": result
    }