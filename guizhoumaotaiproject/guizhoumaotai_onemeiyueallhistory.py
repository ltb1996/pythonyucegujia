import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os

class MaotaiTrendPredictor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        
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
            '成交量': 'Volume',
            '成交额': 'Amount',
            '涨跌幅': 'Change_Pct'
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
        
    def prepare_data(self):
        # 检查数据是否为空
        if len(self.df) == 0:
            raise ValueError("没有数据可用于准备特征")
        
        # 确保所需特征列都存在
        required_features = ['SMA_5', 'SMA_20', 'RSI', 'Close', 'Volume']
        missing_features = [f for f in required_features if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"缺少以下特征列: {missing_features}")
        
        # 准备特征
        X = self.df[required_features].values
        y = self.df['Target'].values
        
        # 打印特征形状
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")
        
        # 数据标准化
        X = self.scaler.fit_transform(X)
        
        return X, y
        
    def train_model(self, X, y):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Return the test set score
        return self.model.score(X_test, y_test)
        
    def predict_future_trend(self, month=6, days=30):
        # 获取最新的收盘价
        last_price = self.df['Close'].iloc[-1]
        
        # 生成6月份的日期序列（1号到30号）
        current_year = datetime.now().year
        future_dates = [datetime(current_year, month, day) for day in range(1, days + 1)]
        
        # 获取最新的特征数据
        last_data = self.df[['SMA_5', 'SMA_20', 'RSI', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
        last_data_scaled = self.scaler.transform(last_data)
        
        # 预测趋势概率
        trend_prob = self.model.predict_proba(last_data_scaled)[0][1]
        
        # 使用更小的波动率来生成更平滑的价格变动
        daily_std = self.df['Close'].pct_change().std() * 0.5
        
        # 生成更平滑的价格序列
        predicted_prices = [last_price]
        current_trend = 1 if trend_prob > 0.5 else -1
        trend_change_points = np.random.choice(range(5, 25), size=4)
        
        # 添加小随机因子以避免重复值
        random_factors = np.random.uniform(-0.0001, 0.0001, days)
        
        for i in range(days):
            if i in trend_change_points:
                current_trend *= -1
            
            # 生成基础波动
            base_return = np.random.normal(0, daily_std)
            
            # 添加趋势影响
            trend_impact = current_trend * daily_std * 0.8
            
            # 计算总回报率（加入小随机因子）
            total_return = base_return + trend_impact + random_factors[i]
            
            # 确保价格变动不会太大
            total_return = np.clip(total_return, -daily_std * 2, daily_std * 2)
            
            # 计算新价格
            new_price = predicted_prices[-1] * (1 + total_return)
            
            # 确保价格不重复
            while any(abs(p - new_price) < 0.01 for p in predicted_prices):
                random_adjustment = np.random.uniform(-0.02, 0.02)
                new_price *= (1 + random_adjustment)
            
            predicted_prices.append(new_price)
        
        predicted_prices = predicted_prices[1:]
        
        # 对于周末的价格，使用稍微调整的前一个交易日价格
        for i, date in enumerate(future_dates):
            if date.weekday() >= 5:  # 周六或周日
                base_price = predicted_prices[i-1]
                small_adjustment = np.random.uniform(-0.02, 0.02)
                predicted_prices[i] = base_price * (1 + small_adjustment)
        
        # 确保最终价格不会偏离太远
        max_change = 0.1
        final_change = (predicted_prices[-1] - last_price) / last_price
        if abs(final_change) > max_change:
            adjustment_factor = max_change / abs(final_change)
            predicted_prices = [last_price + (p - last_price) * adjustment_factor for p in predicted_prices]
            
            # 再次确保没有重复值
            for i in range(1, len(predicted_prices)):
                while any(abs(predicted_prices[j] - predicted_prices[i]) < 0.01 for j in range(i)):
                    predicted_prices[i] *= (1 + np.random.uniform(-0.01, 0.01))
        
        # 保留两位小数，但确保没有重复值
        predicted_prices = [round(price, 2) for price in predicted_prices]
        for i in range(1, len(predicted_prices)):
            while predicted_prices[i] in predicted_prices[:i]:
                predicted_prices[i] = round(predicted_prices[i] + 0.01, 2)
        
        return future_dates, predicted_prices
        
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
            
            save_path = os.path.join(save_dir, 'maotai_trend_prediction_june_allhistory.png')
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # 第一个子图：修改为显示全部历史价格
            all_data = self.df  # 使用全部数据而不是仅使用最近60天
            ax1.plot(all_data['Date'], all_data['Close'], label='历史价格', color='blue', linewidth=2)
            ax1.set_title('贵州茅台股票价格（全部历史）', pad=15, fontproperties='SimHei')
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
            ax2.set_title('贵州茅台股票价格（2025年6月预测）', pad=15, fontproperties='SimHei')
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
    predictor = MaotaiTrendPredictor(r'd:\benkeshengxm\pythonyucegujia\guizhoumaotaiproject\maotai.csv')
    predictor.load_data()
    predictor.create_features()
    
    X, y = predictor.prepare_data()
    score = predictor.train_model(X, y)
    print(f'模型准确率: {score:.4f}')
    
    # 预测6月份的股价
    future_dates, predicted_prices = predictor.predict_future_trend(month=6, days=30)
    predictor.plot_results(future_dates, predicted_prices)
    print('预测图表已保存为 maotai_trend_prediction_june_allhistory.png')
    
    # 输出当前价格
    current_price = predictor.df['Close'].iloc[-1]
    print(f'\n当前价格: {current_price:.2f}')
    
    # 输出6月份每日预测价格
    print("\n2025年6月预测价格:")
    for date, price in zip(future_dates, predicted_prices):
        print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
    
    # 计算6月份整体涨跌幅
    total_change = ((predicted_prices[-1] - current_price) / current_price) * 100
    print(f"\n6月份预测总体涨跌幅: {total_change:.2f}%")

if __name__ == '__main__':
    main()
