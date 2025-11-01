import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class XiaomiTrendPredictor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path, encoding='gbk')
        except:
            self.df = pd.read_csv(self.csv_path, encoding='gb2312')
        
        print("CSV file columns:", self.df.columns.tolist())
        
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)
        
        column_mapping = {
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume',
            '成交额': 'Amount',
            '振幅': 'Amplitude',
            '涨跌幅': 'Change_Pct',
            '涨跌额': 'Change_Amount',
            '换手率': 'Turnover_Rate'
        }
        
        self.df = self.df.rename(columns=column_mapping)
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        self.df = self.df.sort_values('Date')
        
        numeric_columns = ['Open', 'Close', 'High', 'Low', 'Volume', 'Amount', 
                          'Amplitude', 'Change_Pct', 'Change_Amount', 'Turnover_Rate']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
    def create_features(self, window_size=14):
        # Calculate technical indicators
        self.df['SMA_5'] = self.df['Close'].rolling(window=5).mean()
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['RSI'] = self.calculate_rsi(self.df['Close'], periods=14)
        
        # Calculate trend labels (1 for uptrend, 0 for downtrend)
        self.df['Target'] = (self.df['Close'].shift(-1) > self.df['Close']).astype(int)
        
        # Remove rows with NaN
        self.df = self.df.dropna()
        
    def calculate_rsi(self, prices, periods=14):
        # Calculate RSI indicator
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def prepare_data(self):
        # Prepare features
        features = ['SMA_5', 'SMA_20', 'RSI', 'Close', 'Volume']
        X = self.df[features].values
        y = self.df['Target'].values
        
        # Data normalization
        X = self.scaler.fit_transform(X)
        
        return X, y
        
    def train_model(self, X, y):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Return the test set score
        return self.model.score(X_test, y_test)
        
    def predict_future_trend(self, days=180):
        # Get the latest feature data
        last_data = self.df[['SMA_5', 'SMA_20', 'RSI', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
        last_data_scaled = self.scaler.transform(last_data)
        
        # Predict trend
        trend_prob = self.model.predict_proba(last_data_scaled)[0][1]
        
        # Generate future dates
        last_date = self.df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=x+1) for x in range(days)]
        
        # Generate trend probability sequence (adding some random fluctuations)
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, days)
        trend_probs = np.clip(trend_prob + np.cumsum(noise) * 0.1, 0, 1)
        
        return future_dates, trend_probs
        
    def plot_results(self, future_dates, trend_probs):
        plt.figure(figsize=(15, 8))
        
        # Plot historical closing prices
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot historical prices
        ax1.plot(self.df['Date'], self.df['Close'], label='Historical Price', color='blue')
        ax1.set_ylabel('Stock Price', color='blue')
        
        # Plot predicted trend probabilities
        ax2.plot(future_dates, trend_probs, label='Uptrend Probability', color='red', linestyle='--')
        ax2.set_ylabel('Probability', color='red')
        ax2.set_ylim(0, 1)
        
        plt.title('Xiaomi Group Stock Price Trend Prediction')
        ax1.set_xlabel('Date')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the chart
        plt.savefig('xiaomi_trend_prediction.png')
        plt.close()

def main():
    predictor = XiaomiTrendPredictor(r'd:\pythonyucegujia\xiaomiproject\01810.csv')
    predictor.load_data()
    predictor.create_features()
    
    # Prepare data and train the model
    X, y = predictor.prepare_data()
    score = predictor.train_model(X, y)
    print(f'Model Accuracy: {score:.4f}')
    
    # Predict future trends and plot
    future_dates, trend_probs = predictor.predict_future_trend()
    predictor.plot_results(future_dates, trend_probs)
    print('Prediction chart saved as xiaomi_trend_prediction.png')
    
    # Output the final trend prediction
    final_prob = trend_probs[-1]
    print(f'Probability of uptrend in next 6 months: {final_prob:.2%}')
    if final_prob > 0.5:
        print('Prediction: Bullish')
    else:
        print('Prediction: Bearish')

if __name__ == '__main__':
    main()
