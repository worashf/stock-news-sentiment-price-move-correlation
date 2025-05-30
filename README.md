


 📈 Stock News Sentiment with Price Analysis

## 🚀 Overview
This project analyzes financial news headlines and stock price data to explore the relationship between **news sentiment** and **stock performance**. The objective is to quantify sentiment using NLP techniques and correlate it with stock market behavior to generate **predictive insights** and **investment strategies**.

This project was completed as part of **10 Academy: Artificial Intelligence Mastery**, to build mastery in:
- EDA
- Financial Analytics (FA)
- NLP
- Statistics
- ML



## 🧠 Business Objective
Nova Financial Solutions seeks to improve predictive analytics for financial forecasting by leveraging **headline sentiment scores** and **stock price data**. This project aims to:
- Derive sentiment from headlines using NLP.
- Track price movement around article timestamps.
- Recommend actionable trading strategies based on sentiment analysis.

---

## 📊 Dataset Description

**Financial News and Stock Price Integration Dataset (FNSPID):**
| Field     | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| headline  | News title containing financial info (e.g., earnings, price target changes) |
| url       | Link to the full article                                                    |
| publisher | Source of the news                                                          |
| date      | Timestamp of publication (UTC-4)                                            |
| stock     | Stock ticker symbol (e.g., AAPL for Apple)                                  |

---

## 🎯 Learning Goals

- ✅ Set up a Python environment and GitHub repo
- ✅ Perform EDA on text + time-series data
- ✅ Run NLP sentiment analysis on headlines
- ✅ Measure correlations between sentiment & returns
- ✅ Compute technical indicators (RSI, MA, MACD)
- ✅ Communicate findings in a concise report

---

## 🧰 Project Structure

```

├── .github/
│   └── workflows/
│       └── unittests.yml       # GitHub Actions CI pipeline
├── .gitignore                  # Ignore venvs, logs, etc.
├── requirements.txt           # pip dependencies
├── README.md                  # Project overview
├── src/
│          # Main library code
├── notebooks/
│   └── README.md              # Notebook guide
├── tests/
          # Unit tests
└── scripts/

└── README.md   # project guide

````

---

## 📦 Installation

### Using Conda:
```bash
conda create -n sentiment-analysis python=3.10
conda activate sentiment-analysis
pip install -r requirements.txt
````

### Or clone this repo:

```bash
git clone git@github.com:worashf/stock-news-sentiment-price-move-correlation.git
cd stock-news-sentiment-price-move-correlation
```

---

## 🧪 Features & Tasks Completed

* [x] Environment setup with Git, GitHub, Conda
* [x] Headline length statistics
* [x] Publisher frequency analysis
* [x] Time trends & spike analysis
* [x] Sentiment scoring using NLP (VADER, TextBlob)
* [x] Correlation of sentiment with stock returns
* [x] Visualization of findings
* [x] GitHub CI/CD with unit tests

---

## 📸 Screenshot Report

Here is a sample output from our EDA dashboard and sentiment correlation plot:

![Sentiment vs Price Report](visuals/sample-report-screenshot.png)

> *Note: Replace this screenshot with actual output image from your visuals directory.*

---

## 📑 Sample Output (Markdown Report)

**Top Publishers:**

```
1. Reuters
2. Bloomberg
3. Yahoo Finance
```

**Sentiment Trends Example (AAPL):**

* Positive headlines around earnings = average 2.3% gain
* Negative sentiment before major price dips

---

## 🔗 Contact

**Worash Abocherugn**
✉️ [worashup@gmail.com](mailto:worashup@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/worash-abocherugn/)

---

## 📄 License

This project is open source under the MIT License.

---

*Built with ❤️ for educational purposes under the 10 Academy: Artificial Intelligence Mastery Program.*

```

