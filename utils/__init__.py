import warnings
from pathlib import Path
from dotenv import load_dotenv
import sys
import os
import re
import json
import ast
import base64
import pandas as pd
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import fitz

warnings.simplefilter("ignore", UserWarning)

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
dotenv_path = ROOT_PATH / 'config/.env'
load_dotenv(dotenv_path)

GOOGLE_CLOUD_CREADENTIALS = os.getenv("P_GOOGLE_CLOUD_CREADENTIALS")
OPENAI_API_KEY =  os.getenv("GPT_API_KEY")
_path =  os.getenv("_path")