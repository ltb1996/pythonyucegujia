import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class FuxingyiyaoLSTMPredictor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = MinMaxScaler()
        self.model = None
        
    def create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
        
    def build_model(self, seq_length, n_features):
        model = Sequential([
            LSTM(100, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)),
            Dropout(0.2),
            LSTM(100, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def prepare_data(self):
        # 准备特征
        features = self.df[['Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI']].values
        
        # 标准化数据
        scaled_features = self.scaler.fit_transform(features)
        
        # 创建序列数据
        seq_length = 30  # 使用30天的数据预测下一天
        X, y = self.create_sequences(scaled_features, seq_length)
        
        # 分割训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test, seq_length
        
    def train_model(self, X_train, y_train, seq_length):
        # 构建和训练模型
        self.model = self.build_model(seq_length, X_train.shape[2])
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        return history
        
    def predict_future_trend(self, month=6, days=30, year=2025):
        last_sequence = self.df[['Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI']].values[-30:]
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        future_dates = [datetime(year, month, day) for day in range(1, days + 1)]
        predicted_prices = []
        current_sequence = last_sequence_scaled.copy()
        
        # 获取最近的价格波动范围
        recent_volatility = np.std(self.df['Close'].tail(30).pct_change()) * 100
        
        for i in range(days):
            # 基础预测
            next_pred = self.model.predict(current_sequence.reshape(1, 30, 5))
            
            # 添加随机波动
            volatility_factor = 0.5  # 控制波动强度
            random_change = np.random.normal(0, recent_volatility * volatility_factor)
            
            # 根据星期几调整波动
            weekday = future_dates[i].weekday()
            if weekday in [0, 4]:  # 周一和周五波动加大
                random_change *= 1.5
            
            # 添加周期性趋势
            cycle_effect = np.sin(2 * np.pi * i / 10) * recent_volatility  # 10天一个周期
            
            # 合并各种效应
            next_pred[0] *= (1 + (random_change + cycle_effect) / 100)
            
            # 更新序列
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pred[0]
            
            # 转换预测值
            pred_price = self.scaler.inverse_transform(current_sequence[-1].reshape(1, -1))[0][0]
            predicted_prices.append(max(round(pred_price, 2), 0))  # 确保价格非负
        
        return future_dates, predicted_prices
        
    def load_data(self):
        # 尝试多种编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16', 'latin1']
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                print(f"尝试 {encoding} 编码失败，继续尝试其他编码...")
                continue
            except Exception as e:
                print(f"使用 {encoding} 编码时发生错误: {str(e)}")
                continue
        
        print("初始数据行数:", len(self.df))
        print("CSV文件列名:", self.df.columns.tolist())
        
        # 删除包含"数据来源"的行
        self.df = self.df[~self.df['日期'].str.contains('数据来源', na=False)]
        print("删除数据来源行后的行数:", len(self.df))
        
        # 重命名列
        column_mapping = {
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量（万股）': 'Volume',
            '成交额（万元）': 'Amount',
            '涨跌幅(%)': 'Change_Pct'
        }
        
        self.df = self.df.rename(columns=column_mapping)
        
        # 转换数值列
        numeric_columns = ['Open', 'Close', 'High', 'Low', 'Volume', 'Amount', 'Change_Pct']
        for col in numeric_columns:
            # 移除可能的逗号和空格
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.replace(',', '').str.strip()
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print("数值转换后的行数:", len(self.df))
        
        # 转换日期列
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        print("日期转换后的行数:", len(self.df))
        
        # 删除包含空值的行
        self.df = self.df.dropna()
        print("删除空值后的行数:", len(self.df))
        
        # 检查是否有重复的日期
        duplicates = self.df['Date'].duplicated()
        if duplicates.any():
            print(f"发现 {duplicates.sum()} 个重复日期")
            self.df = self.df.drop_duplicates(subset=['Date'])
        
        # 排序数据
        self.df = self.df.sort_values('Date')
        
        # 打印最终的数据形状
        print("最终数据形状:", self.df.shape)
        print("数据日期范围:", self.df['Date'].min(), "至", self.df['Date'].max())
        
        # 打印数据样本
        print("\n数据样本:")
        print(self.df.head())
        
        if len(self.df) == 0:
            raise ValueError("数据处理后没有剩余有效数据，请检查数据源")
        
        print("实际的CSV列名:", list(self.df.columns))
        
    def create_features(self, window_size=14):
        # 确保数据不为空
        if len(self.df) == 0:
            raise ValueError("没有有效的数据用于创建特征")
        
        # 计算技术指标
        self.df['SMA_5'] = self.df['Close'].rolling(window=5).mean()
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['RSI'] = self.calculate_rsi(self.df['Close'], periods=14)
        
        # 计算趋势标签
        self.df['Target'] = (self.df['Close'].shift(-1) > self.df['Close']).astype(int)
        
        # 删除包含NaN的行
        self.df = self.df.dropna()
        
        # 打印特征创建的数据形状
        print(f"特征创建完成，剩余 {len(self.df)} 行数据")
        
    def calculate_rsi(self, prices, periods=14):
        # Calculate RSI indicator
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def plot_results(self, future_dates, predicted_prices):
        print("开始生成预测图表...")
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 创建保存图表的目录
            save_dir = 'prediction_results'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            save_path = os.path.join(save_dir, 'fuxingyiyaolstm_trend_prediction_june_allhistory.png')
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # 第一个子图：修改为显示全部历史价格
            all_data = self.df  # 使用全部数据而不是仅使用最近60天
            ax1.plot(all_data['Date'], all_data['Close'], label='历史价格', color='blue', linewidth=2)
            ax1.set_title('复星医药股票价格（全部历史）', pad=15, fontproperties='SimHei')
            ax1.set_xlabel('日期', fontproperties='SimHei')
            ax1.set_ylabel('股票价格 (元)', fontproperties='SimHei')
            ax1.legend(loc='upper left', prop={'family':'SimHei'})
            
            # 设置x轴日期格式，避免日期标签重叠
            ax1.xaxis.set_major_locator(mdates.YearLocator())  # 每年显示一个刻度
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 显示年份
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 标注最新价格
            last_price = self.df['Close'].iloc[-1]
            last_date = self.df['Date'].iloc[-1]
            ax1.annotate(f'最新价格: {last_price:.2f}', 
                        xy=(last_date, last_price),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->'),
                        fontproperties='SimHei')
            
            # 第二个子图：6月份预测价格
            ax2.plot(future_dates, predicted_prices, 'ro-', label='6月预测', linewidth=2)
            ax2.set_title('复星医药股票价格（2025年6月预测）', pad=15, fontproperties='SimHei')
            ax2.set_xlabel('日期', fontproperties='SimHei')
            ax2.set_ylabel('预测价格 (元)', fontproperties='SimHei')
            
            # 设置x轴日期格式和刻度
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            # 强制显示所有30天的刻度
            ax2.set_xticks(future_dates)
            plt.xticks(rotation=45)
            
            # 设置y轴范围
            min_price = min(predicted_prices) * 0.95
            max_price = max(predicted_prices) * 1.05
            ax2.set_ylim(min_price, max_price)
            
            # 标注价格（每隔几天显示一次）
            for i, (date, price) in enumerate(zip(future_dates, predicted_prices)):
                if i % 2 == 0:  # 每隔一天显示价格
                    ax2.annotate(f'{price:.2f}', 
                            xy=(date, price),
                            xytext=(0, 10), 
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            fontproperties='SimHei')
            
            # 标记周末日期
            for date, price in zip(future_dates, predicted_prices):
                if date.weekday() >= 5:  # 周末
                    ax2.axvline(x=date, color='lightgray', linestyle='--', alpha=0.5)
            
            ax2.legend(loc='upper left', prop={'family':'SimHei'})
            plt.tight_layout()
            
            print(f"正在保存图表到 {save_path} ...")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("图表已成功保存！")
            plt.close()
            
        except Exception as e:
            print(f"生成图表时出错: {str(e)}")
            raise

def main():
    predictor = FuxingyiyaoLSTMPredictor(r'd:\benkeshengxm\pythonyucegujia\fuxingyiyaoproject\fuxing.csv')
    predictor.load_data()
    predictor.create_features()
    
    X_train, X_test, y_train, y_test, seq_length = predictor.prepare_data()
    history = predictor.train_model(X_train, y_train, seq_length)
    
    # 预测2025年6月份的股价
    future_dates, predicted_prices = predictor.predict_future_trend(month=6, days=30, year=2025)
    predictor.plot_results(future_dates, predicted_prices)
    print('预测图表已保存为 fuxingyiyaolstm_trend_prediction_june_allhistory.png')
    
    # 输出当前价格
    current_price = predictor.df['Close'].iloc[-1]
    print(f'\n当前价格: {current_price:.2f}')
    
    # 输出2025年6月份每日预测价格
    print("\n2025年6月预测价格:")
    for date, price in zip(future_dates, predicted_prices):
        print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
    
    # 计算6月份整体涨跌幅
    total_change = ((predicted_prices[-1] - current_price) / current_price) * 100
    print(f"\n6月份预测总体涨跌幅: {total_change:.2f}%")

if __name__ == '__main__':
    main()
