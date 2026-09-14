import pandas as pd
import requests
import yfinance as yf
import numpy as np
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("=" * 60)
print("MODULE 1 HOMEWORK - STOCK MARKET ANALYTICS")
print("=" * 60)

# =========================================================================
# Question 1: S&P 500 Stocks Added to the Index
# =========================================================================
print("\n--- Question 1: S&P 500 Additions ---")

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
try:
    response = requests.get(url, headers=headers)
    dfs = pd.read_html(response.text)
    df_sp = dfs[0]
    
    # Extract year from the 'Date added' column
    df_sp['Year'] = df_sp['Date added'].str.extract(r'(\d{4})').astype(float)
    
    # 1. Additions per year starting from 2020
    recent_additions = df_sp[df_sp['Year'] >= 2020]['Year'].value_counts().sort_index()
    max_year = recent_additions.idxmax()
    max_count = recent_additions.max()
    print(f"Year with highest additions since 2020: {int(max_year)} ({max_count} additions)")
    
    # 2. Stocks in the index for more than 20 years (added before 2006, assuming current year is 2026)
    long_term_stocks = df_sp[df_sp['Year'] <= 2005]
    print(f"Number of current S&P 500 stocks added >20 years ago: {len(long_term_stocks)}")
    
except Exception as e:
    print(f"Error fetching Wikipedia data: {e}")

# =========================================================================
# Question 2: Indexes YTD (as of 21 August 2026)
# =========================================================================
print("\n--- Question 2: Indexes YTD Performance ---")

tickers = {
    'US_S&P500': '^GSPC', 'China_Shanghai': '000001.SS', 'HongKong_HSI': '^HSI',
    'Australia_ASX200': '^AXJO', 'India_Nifty50': '^NSEI', 'Canada_TSX': '^GSPTSE',
    'Germany_DAX': '^GDAXI', 'UK_FTSE100': '^FTSE', 'Japan_Nikkei225': '^N225',
    'Mexico_IPC': '^MXX', 'Brazil_Ibovespa': '^BVSP'
}

results = {}
for name, ticker in tickers.items():
    try:
        # end date is exclusive in yfinance, so we use 2026-08-22 to include Aug 21
        data = yf.Ticker(ticker).history(start='2026-01-01', end='2026-08-22')
        if not data.empty:
            ytd_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
            results[name] = ytd_return
        time.sleep(0.5) # Prevent rate limiting
    except Exception as e:
        print(f"Could not download {ticker}: {e}")

df_results = pd.DataFrame.from_dict(results, orient='index', columns=['YTD_Return_%'])
df_results = df_results.sort_values(by='YTD_Return_%', ascending=False)
print(df_results)

us_return = results.get('US_S&P500', 0)
better_than_us = sum(1 for name, ret in results.items() if name != 'US_S&P500' and ret > us_return)
print(f"\nNumber of indexes with better YTD return than US S&P 500: {better_than_us}")

# =========================================================================
# Question 3: S&P 500 Market Corrections Analysis
# =========================================================================
print("\n--- Question 3: Market Corrections Analysis ---")

periods = [
    ('1950-01-01', '1980-01-01'),
    ('1980-01-01', '2000-01-01'),
    ('2000-01-01', '2020-01-01'),
    ('2020-01-01', '2026-09-14')
]

dfs = []
for start, end in periods:
    try:
        data = yf.download('^GSPC', start=start, end=end, progress=False)
        if not data.empty:
            # Robustly extract the 'Close' series regardless of yfinance version
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data['Close'].iloc[:, 0]
            else:
                close_data = data['Close']
            df = pd.DataFrame({'Close': close_data})
            dfs.append(df)
        time.sleep(1) # Rate limit buffer
    except Exception as e:
        print(f"Error downloading {start} to {end}: {e}")

if dfs:
    df_sp500 = pd.concat(dfs).dropna().sort_index()

    # Identify All-Time Highs (ATH)
    df_sp500['Cumulative_Max'] = df_sp500['Close'].cummax()
    df_sp500['Is_ATH'] = df_sp500['Close'] == df_sp500['Cumulative_Max']
    ath_df = df_sp500[df_sp500['Is_ATH']].copy()

    # Calculate corrections between consecutive ATHs
    corrections = []
    for i in range(len(ath_df) - 1):
        current_ath_date = ath_df.index[i]
        next_ath_date = ath_df.index[i+1]
        current_ath_price = ath_df.loc[current_ath_date, 'Close']
        
        # Get prices between current ATH and the next ATH
        period = df_sp500.loc[current_ath_date:next_ath_date, 'Close']
        min_price = period.min()
        min_date = period.idxmin()
        
        drawdown_pct = (current_ath_price - min_price) / current_ath_price * 100
        duration_days = (min_date - current_ath_date).days
        
        # Filter for corrections >= 5%
        if drawdown_pct >= 5.0:
            corrections.append({
                'Drawdown_%': drawdown_pct,
                'Duration_Days': duration_days
            })

    corr_df = pd.DataFrame(corrections)

    if not corr_df.empty:
        print(f"Total Number of corrections >= 5%: {len(corr_df)}")
        print(f"Median (50th) Drawdown: {np.percentile(corr_df['Drawdown_%'], 50):.2f}%")
        print(f"Median (50th) Duration: {np.percentile(corr_df['Duration_Days'], 50):.0f} days")
        print("\nTop 3 Largest Corrections by Drawdown:")
        print(corr_df.nlargest(3, 'Drawdown_%'))
    else:
        print("No corrections found.")
else:
    print("Failed to download S&P 500 data.")

# =========================================================================
# Question 4: Earnings Surprise Analysis for Amazon (AMZN)
# =========================================================================
print("\n--- Question 4: Amazon (AMZN) Earnings Surprise ---")

ticker = 'AMZN'
ticker_obj = yf.Ticker(ticker)

earnings = ticker_obj.get_earnings_dates()
earnings = earnings.dropna(subset=['Surprise(%)'])

# Make earnings index timezone-naive
earnings.index = pd.to_datetime(earnings.index)
if earnings.index.tz is not None:
    earnings.index = earnings.index.tz_localize(None)

positive_earnings = earnings[earnings['Surprise(%)'] > 0].copy()

# Get historical price data
prices = yf.download(ticker, start='2019-01-01', end='2026-09-14', progress=False)

# Handle yfinance MultiIndex columns if present
if isinstance(prices.columns, pd.MultiIndex):
    prices = prices['Close'].iloc[:, 0].to_frame('Close')
else:
    prices = prices[['Close']].copy()

# Make prices index timezone-naive
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
prices = prices.sort_index()

# Calculate 2-day returns around earnings
two_day_returns = []
surprises = []

for date in positive_earnings.index:
    try:
        # Find the closest trading day strictly BEFORE the earnings date (Day 1)
        past_dates = prices.index[prices.index < date]
        # Find the closest trading day strictly AFTER the earnings date (Day 3)
        future_dates = prices.index[prices.index > date]
        
        if len(past_dates) == 0 or len(future_dates) == 0:
            continue
            
        day1_date = past_dates[-1]
        day3_date = future_dates[0]
        
        day1_price = prices.loc[day1_date, 'Close']
        day3_price = prices.loc[day3_date, 'Close']
        
        # Calculate 2-day return
        ret_2day = (day3_price / day1_price) - 1
        surprise = positive_earnings.loc[date, 'Surprise(%)'] / 100
        
        two_day_returns.append(ret_2day)
        surprises.append(surprise)
        
    except Exception:
        continue

analysis_df = pd.DataFrame({'Surprise': surprises, 'Return_2Day': two_day_returns})

if not analysis_df.empty:
    print(f"Number of positive earnings surprises analyzed: {len(analysis_df)}")
    print(f"Median 2-Day Return for Positive Surprises: {analysis_df['Return_2Day'].median():.2%}")
    
    correlation = analysis_df.corr().iloc[0, 1]
    print(f"Correlation between Surprise and Return: {correlation:.4f}")
else:
    print("No data could be matched. Check date alignments.")

print("\n" + "=" * 60)
print("Script execution complete.")
print("=" * 60)
