import pandas as pd
import numpy as np
import yfinance as yf
import math
import os
from scipy.optimize import minimize
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#import the black litterman portfolio python file
import black_litterman.black_litterman_portfolio_12months_rolling as blo
