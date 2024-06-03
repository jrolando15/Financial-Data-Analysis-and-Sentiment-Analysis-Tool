# Financial-Data-Analysis-and-Sentiment-Analysis-Tool

# Project Description
This project aims to build a comprehensive tool for financial data analysis and sentiment analysis. It involves extracting financial reports, fetching news articles, performing sentiment analysis on the news, and analyzing stock data. The project uses various Python libraries to process and analyze the data, providing insights into financial performance and public sentiment.

# Table of Content
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Financial Analysis](#financial-analysis)
- [Sentiment Analysis](#sentiment-analysis)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

# Installation
To run this project, you need to have Python installed along with the following libraries:
- selenium
- pandas
- yfinance
- gnews
- nltk
- scikit-learn
- matplotlib
- dateparser
You can install the required libraries using the following command:
```bash
pip install selenium pandas yfinance gnews nltk scikit-learn matplotlib dateparser
```

# Usage
1. Clone the repository
```bash
git clone https://github.com/your_username/Financial-Data-Analysis-and-Sentiment-Analysis-Tool.git
cd Financial-Data-Analysis-and-Sentiment-Analysis-Tool
```

2. Run the main Script
```bash
model.ipynb
```

# Project Structure
```bash
Financial-Data-Analysis-and-Sentiment-Analysis-Tool/
├── model.ipynb                       # Main Jupyter Notebook with the code
├── financial_reports/                # Directory to store financial reports
│   ├── AAPL_balance_sheet.csv
│   ├── AAPL_income_statement.csv
│   ├── AAPL_cash_flow.csv
│   ├── AMZN_balance_sheet.csv
│   ├── AMZN_income_statement.csv
│   ├── AMZN_cash_flow.csv
├── news/                             # Directory to store news articles
│   ├── Apple_news.csv
│   ├── Amazon_news.csv
├── financial_reports_cleaned/        # Directory to store cleaned financial reports
│   ├── Apple_balance_sheet_cleaned.csv
│   ├── Apple_income_statement_cleaned.csv
│   ├── Apple_cash_flow_cleaned.csv
│   ├── Amazon_balance_sheet_cleaned.csv
│   ├── Amazon_income_statement_cleaned.csv
│   ├── Amazon_cash_flow_cleaned.csv
├── README.md                         # Project README file
```

# Data Processing
1. Cleaning the Data
```bash
def clean_data(df):
    df = df.drop_duplicates()
    df = df.ffill().bfill()
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

amazon_data = {key : clean_data(df) for key, df in amazon_data.items()}
apple_data = {key: clean_data(df) for key, df in apple_data.items()}
```

2. Normalizing the Data
```bash
def normalize_data(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

amazon_data = {key: normalize_data(df) for key, df in amazon_data.items()}
apple_data = {key: normalize_data(df) for key, df in apple_data.items()}
```

3. Saving Cleaned Data
```bash
def save_data(data_dict, prefix):
    for key, df in data_dict.items():
        df.to_csv(f'C:\\Users\\Lenovo\\Documents\\Projects\\web_scraping_model2.1\\financial_reports_cleaned\\{prefix}_{key}_cleaned.csv', index=False)

save_data(amazon_data, 'Amazon')
save_data(apple_data, 'Apple'
```

# Financial Analysis
1. First, we need to extract the relevant financial data from the balance sheet and income statement.
```python
def extract_data(balance_sheet, income_statement, year):
    year_date = pd.to_datetime(year, format='%Y-%m-%d', errors='coerce')
    if year_date not in balance_sheet.index or year_date not in income_statement.index:
        raise KeyError(f"Date {year_date} not found in DataFrame index.")
    
    current_assets = balance_sheet.loc[year_date, 'Current Assets']
    current_liabilities = balance_sheet.loc[year_date, 'Current Liabilities']
    total_liabilities = balance_sheet.loc[year_date, 'Total Liabilities Net Minority Interest']
    stockholders_equity = balance_sheet.loc[year_date, 'Stockholders Equity']
    net_income = income_statement.loc[year_date, 'Net Income']
    shares_outstanding = income_statement.loc[year_date, 'Diluted Average Shares']
    
    return current_assets, current_liabilities, total_liabilities, stockholders_equity, net_income, shares_outstanding
```

2. Calculating Financial Ratios
```python
def calculate_ratios(balance_sheet, income_statement, year):
    try:
        current_assets, current_liabilities, total_liabilities, stockholders_equity, net_income, shares_outstanding = extract_data(balance_sheet, income_statement, year)
        
        pe_ratio = net_income / shares_outstanding
        debt_to_equity = total_liabilities / stockholders_equity
        roe = net_income / stockholders_equity
        
        return {
            'PE_Ratio': pe_ratio,
            'Debt_to_Equity': debt_to_equity,
            'ROE': roe
        }
    except KeyError as e:
        print(f"Skipping calculation for {year}: {e}")
        return None
```

3. Calculation Ratios for Multiple Years
```python
# List of years to calculate ratios for
years = ['2023-09-30', '2022-09-30', '2021-09-30', '2020-09-30']

# Calculate ratios for Amazon
amzn_ratios = {year: calculate_ratios(amzn_balance_sheet, amzn_income_statement, year) for year in years if calculate_ratios(amzn_balance_sheet, amzn_income_statement, year) is not None}

# Calculate ratios for Apple
aapl_ratios = {year: calculate_ratios(aapl_balance_sheet, aapl_income_statement, year) for year in years if calculate_ratios(aapl_balance_sheet, aapl_income_statement, year) is not None}
```

4. Converting Ratios into Dataframe
To facilitate analysis and visualization, we convert the calculated ratios into a DataFrame.
```python
# Function to convert ratios to DataFrame
def ratios_to_dataframe(ratios):
    df = pd.DataFrame(ratios).T
    df.index.name = 'Year'
    return df

# Convert ratios to DataFrame for Amazon and Apple
amzn_ratios_df = ratios_to_dataframe(amzn_ratios)
aapl_ratios_df = ratios_to_dataframe(aapl_ratios)
```

5. Displaying Financial Ratios
```python
# Function to display ratios
def display_ratios(ratios_df, company_name):
    print(f"Financial Ratios for {company_name}:")
    print(ratios_df)

# Display ratios for Amazon
display_ratios(amzn_ratios_df, 'Amazon')

# Display ratios for Apple
display_ratios(aapl_ratios_df, 'Apple')
```
6. Comparing Performance
```python
def compare_performance(ratios_df, company_name):
    ratios_df.plot(kind='bar', figsize=(14,7))
    plt.title(f'Financial Ratios Comparison for {company_name}')
    plt.xlabel('Year')
    plt.ylabel('Ratio Value')
    plt.legend(loc='best')
    plt.show()

compare_performance(amzn_ratios_df, 'Amazon')
compare_performance(aapl_ratios_df, 'Apple')
```

# Sentiment Analysis
```python
def fetch_news(ticker, start_date, end_date, limit=100):
    google_news = GNews()
    search = google_news.get_news(ticker)
    
    news_data = []
    for article in search[:limit]:
        headline = article.get('title', 'No Title')
        link = article.get('url', 'No URL')
        summary = article.get('description', 'No Description')
        published_date_str = article.get('published date', None)
        
        if published_date_str:
            published_date = dateparser.parse(published_date_str)
            if published_date:
                published_date = published_date.date()
            else:
                published_date = 'No Date'
        else:
            published_date = 'No Date'
        
        news_data.append({
            'headline': headline,
            'summary': summary,
            'link': link,
            'published_date': published_date
        })
    
    if news_data:
        keys = news_data[0].keys()
        save_dir = r"C:\Users\Lenovo\Documents\Projects\web_scraping_model2.1\news"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'{ticker}_news.csv')
        with open(file_path, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(news_data)
    else:
        print(f'No news data found for {ticker}')
        
    return news_data

# Fetch news for Amazon and Apple
amzn_news = fetch_news("Amazon", "2010-01-01", "2024-01-01")
aapl_news = fetch_news("Apple", "2010-01-01", "2024-01-01")

# Convert to DataFrame for better visualization
amzn_news_df = pd.DataFrame(amzn_news)
aapl_news_df = pd.DataFrame(aapl_news)

print("Amazon News")
print(amzn_news_df)

print("\nApple News")
print(aapl_news_df)
```

2. Performing Sentiment Analysis
Sentiment analysis is performed on the news articles using the nltk.sentiment library.
```python
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return 'Neutral'

# Define the paths to the CSV files
amzn_news_path = r"C:\Users\Lenovo\Documents\Projects\web_scraping_model2.1\news\Amazon_news.csv"
aapl_news_path = r"C:\Users\Lenovo\Documents\Projects\web_scraping_model2.1\news\Apple_news.csv"

# Read the CSV files into DataFrames
amzn_news_df = pd.read_csv(amzn_news_path)
aapl_news_df = pd.read_csv(aapl_news_path)

# Apply sentiment analysis to the 'summary' column and create a new 'Sentiment' column
amzn_news_df['Sentiment'] = amzn_news_df['summary'].apply(analyze_sentiment)
aapl_news_df['Sentiment'] = aapl_news_df['summary'].apply(analyze_sentiment)

# Save the updated DataFrames back to the CSV files
amzn_news_df.to_csv(amzn_news_path, index=False)
aapl_news_df.to_csv(aapl_news_path, index=False)

# Print the sentiment analysis results
print("Amazon News Sentiment Analysis:")
print(amzn_news_df['Sentiment'].value_counts())

print("\nApple News Sentiment Analysis:")
print(aapl_news_df['Sentiment'].value_counts())
```

3.  Visualing Sentiment Analysis Results
```python
def plot_sentiment_distribution(df, company_name):
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', figsize=(10, 6), color=['green', 'red', 'blue'])
    plt.title(f'Sentiment Distribution for {company_name}')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

# Plot sentiment distribution for Amazon and Apple
plot_sentiment_distribution(amzn_news_df, 'Amazon')
plot_sentiment_distribution(aapl_news_df, 'Apple')
```

# Results
1. Stock Calculation AAPL
![image](https://github.com/jrolando15/Financial-Data-Analysis-and-Sentiment-Analysis-Tool/assets/124316952/422e6c0c-068f-43e3-a929-0d294f039c6e)

2. Stock Calculation AMZN
![image](https://github.com/jrolando15/Financial-Data-Analysis-and-Sentiment-Analysis-Tool/assets/124316952/c5e858ab-acbf-462a-9e1b-83e79411007e)

3. Financial Results
```bash
Financial Ratios for Amazon:
            PE_Ratio  Debt_to_Equity       ROE
Year                                          
2023-09-30 -0.438429        4.059206  3.725494
2022-09-30 -1.507855       -0.652745 -0.373396
2021-09-30  1.182789        1.459354  1.329122
2020-09-30 -1.225358       -1.603761 -2.341944
Financial Ratios for Apple:
            PE_Ratio  Debt_to_Equity        ROE
Year                                           
2023-09-30  0.407381        0.708224   0.455530
2022-09-30  2.444380        3.382074  -6.490543
2021-09-30  4.815411        1.623545  10.944048
2020-09-30 -0.069753        1.120385  -0.044986
```

4. Sentiment Analysis Results
```bash
Amazon News Sentiment Analysis:
Sentiment
Positive    76
Negative    18
Neutral      6
Name: count, dtype: int64

Apple News Sentiment Analysis:
Sentiment
Neutral     43
Positive    33
Negative    24
Name: count, dtype: int64
```

# Future Work
<div style="text-align: justify;">
  In the future, I intend to improve the project by developing a user-friendly application that allows users to input information for a specific company. The app will provide the following features:
  <ul>
    <li><strong>Financial Information:</strong> Retrieve and display detailed financial reports for the specified company.</li>
    <li><strong>Stock Prices:</strong> Fetch and visualize historical and current stock prices for the company.</li>
    <li><strong>News:</strong> Aggregate and analyze news articles related to the company, with sentiment analysis to gauge public opinion.</li>
    <li><strong>Investment Plan:</strong> Generate an investment plan based on the financial information, stock prices, and news sentiment. The app will provide recommendations on whether it is a good idea to invest in the company.</li>
  </ul>
  By integrating these features, the app will offer comprehensive insights and assist users in making informed investment decisions.
</div>

<div style="text-align: justify;">
  Any inquiries about working together on this project, you can reach out to me through jrolandocollege@gmail.com.
</div>

# License
This README template includes all the pertinent information about your project, such as installation instructions, usage, project structure, data processing, model training, model evaluation, and details about the web application. It also includes sections for contributing and licensing, which are important for open-source projects.
