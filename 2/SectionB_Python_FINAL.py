
# DRM Project – Section B Python Script

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Download TECHM data
data=yf.download("TECHM.NS",start="2025-03-04",end="2026-03-04")
spot=data['Close']

# Basic statistics
returns=spot.pct_change().dropna()
print("Mean:",returns.mean())
print("Std Dev:",returns.std())

# Pricing parameters
r=0.054
d=0
c=0
y=0

expiry={
"Feb2026":datetime(2026,2,26),
"Mar2026":datetime(2026,3,26),
"Feb2027":datetime(2027,2,4)
}

futures=pd.DataFrame(index=spot.index)

for k,v in expiry.items():
    T=(v-spot.index).days/365
    T=np.maximum(T,0)
    futures[k]=spot*np.exp((r-d+c-y)*T)

futures.to_csv("theoretical_futures.csv")

# NOTE:
# Full comparison with actual futures prices and weekly RBI rates
# is implemented in the Excel file SectionB_FINAL_Submission.xlsx
