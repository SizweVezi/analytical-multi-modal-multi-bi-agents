import sys
import os
import boto3
import json
import requests
from datetime import date, datetime
from pprint import pprint
from botocore.client import Config
from botocore.exceptions import ClientError
import pandas as pd
import random
import base64
import io
import matplotlib.pyplot as plt


from typing import List, TypedDict, Any, Tuple, Union, Dict, Set, Annotated, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws import BedrockEmbeddings


from PIL import Image

from chromadb import Documents, EmbeddingFunction, Embeddings
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price

import operator
import re

from utils.knowledge_base_operators import (
    extract_audio_path_and_timestamps_agent_response,
    extract_audio_path_and_timestamps,
    play_audio_segments_from_s3
)
