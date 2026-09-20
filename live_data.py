from fredapi import Fred
import pandas as pd
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read API key securely
FRED_API_KEY = os.getenv("FRED_API_KEY")

fred = Fred(api_key=FRED_API_KEY)

term_spread = fred.get_series("T10Y3M")

df = pd.DataFrame(term_spread, columns=["Term_Spread"])
df.index.name = "Date"

print(df.tail())

