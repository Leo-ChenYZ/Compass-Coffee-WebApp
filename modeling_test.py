import pandas as pd
import numpy as np
from modeling import run_forecast

# Mock preprocessed data — mimics what preprocessing.py would output
df = pd.read_csv("mock_preprocessed_data.csv") 

# Run forecast
result = run_forecast(df)
print(result)