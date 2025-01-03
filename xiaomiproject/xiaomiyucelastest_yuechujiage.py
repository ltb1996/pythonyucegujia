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
        
    def predict_future_trend(self, months=6):
        # 获取最新的收盘价
        last_price = self.df['Close'].iloc[-1]
        
        # 获取最新的特征数据
        last_data = self.df[['SMA_5', 'SMA_20', 'RSI', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
        last_data_scaled = self.scaler.transform(last_data)
        
        # 预测趋势概率
        trend_prob = self.model.predict_proba(last_data_scaled)[0][1]
        
        # 生成2025年1月到6月的日期
        future_dates = []
        start_year = 2025
        for month in range(1, 7):  # 1到6月
            month_start = datetime(start_year, month, 1)
            future_dates.append(month_start)
        
        # 基于历史数据计算参数
        monthly_std = self.df['Close'].pct_change().std() * np.sqrt(21)  # 月度波动率
        monthly_mean = self.df['Close'].pct_change().mean() * 21  # 月度平均收益率
        
        np.random.seed(42)
        
        # 生成更自然的价格序列
        predicted_prices = [last_price]
        
        # 生成基础趋势
        base_trend = (trend_prob - 0.5) * 2 * monthly_std
        
        for i in range(months):
            # 综合多个因素生成价格变化
            
            # 1. 趋势因子 (考虑整体趋势)
            trend_factor = base_trend + monthly_mean
            
            # 2. 周期性因子 (添加市场周期性波动)
            cycle_factor = np.sin(2 * np.pi * i / 6) * monthly_std * 0.5
            
            # 3. 随机波动因子 (模拟市场随机性)
            random_factor = np.random.normal(0, monthly_std * 0.3)
            
            # 4. 动量因子 (考虑前期走势的影响)
            momentum_factor = 0
            if i > 0:
                prev_return = (predicted_prices[-1] / predicted_prices[-2] - 1)
                momentum_factor = prev_return * 0.3
            
            # 5. 均值回归因子 (防止价格偏离过远)
            reversion_factor = (last_price - predicted_prices[-1]) * 0.1
            
            # 合并所有因子
            total_return = (trend_factor + 
                           cycle_factor + 
                           random_factor + 
                           momentum_factor + 
                           reversion_factor)
            
            # 限制单月最大变化幅度
            max_monthly_change = 0.12  # 12%
            total_return = np.clip(total_return, -max_monthly_change, max_monthly_change)
            
            # 计算新价格
            next_price = predicted_prices[-1] * (1 + total_return)
            
            # 确保价格不会低于历史最低价的50%或高于历史最高价的150%
            hist_min = self.df['Close'].min() * 0.5
            hist_max = self.df['Close'].max() * 1.5
            next_price = np.clip(next_price, hist_min, hist_max)
            
            predicted_prices.append(next_price)
        
        predicted_prices = predicted_prices[1:]  # 移除初始价格
        
        return future_dates, predicted_prices
        
    def plot_results(self, future_dates, predicted_prices):
        # 创建一个包含两个子图的图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 第一个子图：历史价格
        ax1.plot(self.df['Date'], self.df['Close'], label='Historical Price', color='blue', linewidth=2)
        ax1.set_title('Xiaomi Historical Stock Price', pad=15)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price (HKD)')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        # 获取最近的价格数据用于标注
        last_price = self.df['Close'].iloc[-1]
        last_date = self.df['Date'].iloc[-1]
        ax1.annotate(f'Last Price: {last_price:.2f}', 
                    xy=(last_date, last_price),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->'))
        
        # 第二个子图：月度预测价格
        ax2.plot(future_dates, predicted_prices, 'ro-', label='Monthly Prediction', linewidth=2)
        ax2.set_title('Xiaomi Stock Price Prediction (Next 6 Months)', pad=15)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Predicted Price (HKD)')
        ax2.grid(True)
        
        # 设置y轴范围，确保所有价格点��在视图内
        min_price = min(predicted_prices) * 0.95  # 留出5%的边距
        max_price = max(predicted_prices) * 1.05
        ax2.set_ylim(min_price, max_price)
        
        # 在每个预测点添加价格标注
        for i, (date, price) in enumerate(zip(future_dates, predicted_prices)):
            ax2.annotate(f'{price:.2f}', 
                        xy=(date, price),
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        ax2.legend(loc='upper left')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('xiaomi_trend_prediction_yuechu.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    predictor = XiaomiTrendPredictor(r'd:\benkeshengxm\pythonyucegujia\xiaomiproject\01810.csv')
    predictor.load_data()
    predictor.create_features()
    
    # 准备数据并训练模型
    X, y = predictor.prepare_data()
    score = predictor.train_model(X, y)
    print(f'Model Accuracy: {score:.4f}')
    
    # 预测未来价格并绘图
    future_dates, predicted_prices = predictor.predict_future_trend()
    predictor.plot_results(future_dates, predicted_prices)
    print('Prediction chart saved as xiaomi_trend_prediction_yuechu.png')
    
    # 输出当前价格
    current_price = predictor.df['Close'].iloc[-1]
    print(f'\nCurrent Price: {current_price:.2f}')
    
    # 输出每个月的预测价格
    print("\nPredicted Prices for Each Month:")
    for date, price in zip(future_dates, predicted_prices):
        print(f"{date.strftime('%Y-%m')}: {price:.2f}")
    
    # 计算整体涨跌幅
    total_change = ((predicted_prices[-1] - current_price) / current_price) * 100
    print(f"\nOverall Predicted Change: {total_change:.2f}%")

if __name__ == '__main__':
    main()
