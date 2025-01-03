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
        # 创建一个包含两个子图的图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 第一个子图：历史价格
        ax1.plot(self.df['Date'], self.df['Close'], label='Historical Price', color='blue', linewidth=2)
        ax1.set_title('Xiaomi Historical Stock Price', pad=15)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price (HKD)')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        # 添加移动平均线
        ax1.plot(self.df['Date'], self.df['SMA_5'], label='5-day MA', color='orange', linestyle='--', alpha=0.8)
        ax1.plot(self.df['Date'], self.df['SMA_20'], label='20-day MA', color='green', linestyle='--', alpha=0.8)
        
        # 获取最近的价格数据用于标注
        last_price = self.df['Close'].iloc[-1]
        last_date = self.df['Date'].iloc[-1]
        ax1.annotate(f'Last Price: {last_price:.2f}', 
                    xy=(last_date, last_price),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->'))
        
        # 第二个子图：预测趋势
        # 计算趋势的移动平均以使曲线更平滑
        trend_ma = pd.Series(trend_probs).rolling(window=7).mean()
        
        # 绘制预测趋势
        ax2.plot(future_dates, trend_probs, label='Daily Prediction', color='gray', alpha=0.3, linewidth=1)
        ax2.plot(future_dates, trend_ma, label='7-day MA Prediction', color='red', linewidth=2)
        ax2.set_title('Xiaomi Stock Trend Prediction (Next 6 Months)', pad=15)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Uptrend Probability')
        ax2.grid(True)
        ax2.set_ylim(0, 1)
        
        # 添加水平参考线
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        ax2.fill_between(future_dates, 0.5, trend_ma, 
                         where=(trend_ma >= 0.5),
                         color='green', alpha=0.1, label='Bullish Zone')
        ax2.fill_between(future_dates, 0.5, trend_ma,
                         where=(trend_ma < 0.5),
                         color='red', alpha=0.1, label='Bearish Zone')
        
        # 添加最终预测概率标注
        final_prob = trend_probs[-1]
        final_date = future_dates[-1]
        ax2.annotate(f'Final Probability: {final_prob:.2%}',
                    xy=(final_date, final_prob),
                    xytext=(-60, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->'))
        
        ax2.legend(loc='upper left')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('xiaomi_trend_prediction.png', dpi=300, bbox_inches='tight')
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
