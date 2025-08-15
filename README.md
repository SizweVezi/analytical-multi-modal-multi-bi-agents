# BI Analytical Multi Modal Agentic Workflow

A sophisticated multi-modal, multi-agent orchestration pipeline that integrates Amazon Bedrock's managed foundation model hosting, the Nova model family, and LangGraph for intelligent agent coordination and workflow management.

## Overview

This project implements an advanced business intelligence analytical system that combines multiple AI agents to handle diverse query types including audio earnings call analysis, image-based financial data retrieval, company financial analysis, and automated report generation.

## Architecture

The system uses a **router-based agent architecture** with specialized nodes:

- **Router Agent**: Classifies incoming queries and routes them to appropriate specialized agents
- **Audio RAG Agent**: Processes earnings call audio data using Amazon Knowledge Bases
- **Image RAG Agent**: Handles image-based financial diagram and chart queries
- **Financial Analysis Agent**: Performs technical analysis using real-time stock data
- **Report Generation Agent**: Creates comprehensive financial reports
- **LLM Chat Agent**: Handles general conversational queries

## Features

### 🎯 Multi-Modal Intelligence
- **Audio Processing**: Analyze earnings call transcripts with timestamp extraction
- **Image Analysis**: Search and retrieve financial charts, diagrams, and visual data
- **Text Analysis**: Process financial documents and generate insights
- **Video Support**: Handle video content with segment-based playback

### 📊 Financial Analysis Tools
- Real-time stock price monitoring
- Technical indicator calculations (RSI, MACD, SMA, EMA)
- Company financial metrics retrieval
- Market volatility analysis
- Fundamental analysis capabilities

### 🤖 AI-Powered Agents
- **Claude 3.5 Sonnet**: Advanced reasoning and analysis
- **Amazon Nova Pro/Lite**: Multi-modal understanding
- **Mistral Large**: Specialized financial analysis
- **Amazon Titan**: Text embeddings and search

### 🔄 Workflow Orchestration
- LangGraph-based state management
- Conditional routing based on query type
- Tool integration with automatic fallbacks
- Memory persistence across conversations

## Project Structure

```
bi-analytical-multi-modal-agentic-workflow/
├── src/
│   ├── agents/
│   │   └── agentic_workflow.py      # Main workflow orchestration
│   ├── audio-video-rag/
│   │   └── audio-image.py           # Audio/video processing setup
│   ├── config/
│   │   └── settings.py              # Configuration management
│   ├── finance_tools/
│   │   └── stock_ta_analysis.py     # Financial analysis tools
│   ├── utils/
│   │   ├── api_helpers.py           # API and model configurations
│   │   ├── bedrock.py               # AWS Bedrock utilities
│   │   ├── knowledge_base.py        # Knowledge base operations
│   │   └── knowledge_base_operator.py # Advanced KB operations
│   └── main.py                      # Core imports and dependencies
├── tests/                           # Test suite
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites
- Python 3.8+
- AWS Account with Bedrock access
- AWS CLI configured
- Required AWS services: Bedrock, Knowledge Bases, S3

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bi-analytical-multi-modal-agentic-workflow
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure
   ```

4. **Set environment variables**
   ```bash
   export AUDIO_KB_ID="your-audio-knowledge-base-id"
   export IMAGE_KB_ID="your-image-knowledge-base-id"
   export FINANCIAL_MODEL_PREP_KEY="your-fmp-api-key"
   ```

## Configuration

### AWS Bedrock Models
The system supports multiple foundation models:
- `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `us.amazon.nova-pro-v1:0`
- `us.amazon.nova-lite-v1:0`
- `mistral.mistral-large-2402-v1:0`
- `amazon.titan-embed-text-v2:0`

### Knowledge Bases
Configure Amazon Knowledge Bases for:
- **Audio KB**: Earnings call transcripts and audio metadata
- **Image KB**: Financial charts, diagrams, and visual data

## Usage

### Basic Query Processing

```python
from src.agents.agentic_workflow import app

# Configure thread
thread = {"configurable": {"thread_id": "42", "recursion_limit": 10}}

# Process query
result = app.invoke({'question': "Analyze Amazon's Q3 2024 earnings"}, thread)
print(result['answer'])
```

### Query Types

1. **Audio Earnings Analysis**
   ```python
   query = "Give me a summary of Amazon's Q3 2024 earning based on the earning call audio"
   ```

2. **Image-Based Analysis**
   ```python
   query = "Show me diagrams of Amazon TTM operation income and net sales in 2024"
   ```

3. **Stock Analysis**
   ```python
   query = "How about Uber's stock performance?"
   ```

4. **Report Generation**
   ```python
   query = "Create a financial report based on Amazon latest results"
   ```

### Advanced Features

#### Audio Segment Playback
```python
from src.utils.knowledge_base_operators import (
    extract_audio_path_and_timestamps_agent_response,
    play_audio_segments_from_s3
)

# Extract audio information from response
audio_s3_info, timestamps = extract_audio_path_and_timestamps_agent_response(results)

# Play specific segments
play_audio_segments_from_s3(audio_s3_info, timestamps)
```

#### Financial Analysis Tools
```python
from src.finance_tools.stock_ta_analysis import get_stock_prices, get_financial_metrics

# Get real-time stock data
stock_data = get_stock_prices("AAPL")
financial_metrics = get_financial_metrics("AAPL")
```

## API Integration

### Supported APIs
- **Financial Modeling Prep**: Company financials and market data
- **Yahoo Finance**: Real-time stock prices and historical data
- **DuckDuckGo Search**: Web search capabilities
- **AWS Bedrock**: Foundation model inference
- **Amazon Knowledge Bases**: Document retrieval and search

## Workflow Details

### Router Logic
The system automatically classifies queries into categories:
- `audioearningcall`: Audio-based earnings analysis
- `Image_Search`: Visual financial data queries
- `company_financial`: Stock and financial analysis
- `financial_report`: Report generation
- `chat`: General conversational queries

### State Management
LangGraph manages conversation state including:
- Query classification results
- Retrieved documents and citations
- Audio/video metadata and timestamps
- Generated responses and tool outputs
- Error handling and recovery

## Security & Best Practices

- **Credential Management**: Use environment variables for API keys
- **AWS IAM**: Implement least-privilege access policies
- **Data Privacy**: Sanitize PII from responses
- **Rate Limiting**: Built-in retry mechanisms for API calls
- **Error Handling**: Comprehensive exception management

## Performance Optimization

- **Model Selection**: Automatic model routing based on query complexity
- **Caching**: Response caching for repeated queries
- **Parallel Processing**: Concurrent tool execution where possible
- **Resource Management**: Efficient memory usage for large documents

## Troubleshooting

### Common Issues

1. **Knowledge Base Access**
   - Verify KB IDs are correctly set
   - Check IAM permissions for Bedrock and Knowledge Bases

2. **Model Access**
   - Ensure Bedrock model access is enabled in your AWS account
   - Verify region availability for specific models

3. **API Rate Limits**
   - Monitor CloudWatch for throttling events
   - Implement exponential backoff (already included)

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AWS Bedrock team for foundation model access
- LangChain/LangGraph for agent orchestration framework
- Financial Modeling Prep for financial data API
- Open source community for various Python libraries

---

**Note**: This system requires appropriate AWS permissions and API access. Ensure compliance with your organization's data governance policies when processing financial information.