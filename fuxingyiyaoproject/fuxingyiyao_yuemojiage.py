import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class FuxingyiyaoTrendPredictor:
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
        
        # 打印特征创建后的数据形状
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
        
    def predict_future_trend(self, months=6):
        # 获取最新的收盘价
        last_price = self.df['Close'].iloc[-1]
        
        # 获取最新的特征数据
        last_data = self.df[['SMA_5', 'SMA_20', 'RSI', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
        last_data_scaled = self.scaler.transform(last_data)
        
        # 预测趋势概率
        trend_prob = self.model.predict_proba(last_data_scaled)[0][1]
        
        # 生成未来月份的日期（每月最后一天）
        last_date = self.df['Date'].iloc[-1]
        future_dates = []
        for i in range(1, months + 1):
            # 获取下i个月的最后一天
            if last_date.month + i > 12:
                year = last_date.year + (last_date.month + i - 1) // 12
                month = (last_date.month + i - 1) % 12 + 1
            else:
                year = last_date.year
                month = last_date.month + i
            # 使用下个月的第一天减一天来获取当月最后一天
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            month_end = next_month - timedelta(days=1)
            future_dates.append(month_end)
        
        # 基于历史波动性生成月度价格预测
        monthly_std = self.df['Close'].pct_change().std() * np.sqrt(21)  # 月度波动率
        np.random.seed(42)
        monthly_returns = np.random.normal(0, monthly_std, months)
        
        # 根据预测趋势调整价格走向
        trend_adjustment = (trend_prob - 0.5) * 2  # 将概率转换为[-1, 1]范围的调整因子
        monthly_returns += trend_adjustment * monthly_std
        
        # 计算预测价格
        predicted_prices = [last_price]
        for ret in monthly_returns:
            predicted_prices.append(predicted_prices[-1] * (1 + ret))
        predicted_prices = predicted_prices[1:]  # 移除初始价格
        
        return future_dates, predicted_prices
        
    def plot_results(self, future_dates, predicted_prices):
        print("开始生成预测图表...")
        try:
            # 创建一个包含两个子图的图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # 第一个子图：历史价格
            ax1.plot(self.df['Date'], self.df['Close'], label='Historical Price', color='blue', linewidth=2)
            ax1.set_title('fuxingyiyao Stock Price', pad=15, fontproperties='SimHei')
            ax1.set_xlabel('Date', fontproperties='SimHei')
            ax1.set_ylabel('Stock Price (HKD)', fontproperties='SimHei')
            ax1.legend(loc='upper left', prop={'family':'SimHei'})
            
            # 获取最近的价格数据用于标注
            last_price = self.df['Close'].iloc[-1]
            last_date = self.df['Date'].iloc[-1]
            ax1.annotate(f'Last Price: {last_price:.2f}', 
                        xy=(last_date, last_price),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->'),
                        fontproperties='SimHei')
            
            # 第二个子图：月度预测价格
            ax2.plot(future_dates, predicted_prices, 'ro-', label='Monthly Prediction', linewidth=2)
            ax2.set_title('fuxingyiyao Stock Price (Next 6 Months)', pad=15, fontproperties='SimHei')
            ax2.set_xlabel('Date', fontproperties='SimHei')
            ax2.set_ylabel('Predicted Price (HKD)', fontproperties='SimHei')
            
            # 设置y轴范围，确保所有价格点都在视图内
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
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            fontproperties='SimHei')
            
            ax2.legend(loc='upper left', prop={'family':'SimHei'})
            
            # 调整布局
            plt.tight_layout()
            
            print("正在保存图表...")
            plt.savefig('fuxingyiyao_trend_prediction_yuemo.png', dpi=300, bbox_inches='tight')
            print("图表已成功保存！")
            plt.close()
        except Exception as e:
            print(f"生成图表时出错: {str(e)}")

def main():
    predictor = FuxingyiyaoTrendPredictor(r'd:\benkeshengxm\pythonyucegujia\fuxingyiyaoproject\fuxing.csv')
    predictor.load_data()
    predictor.create_features()
    
    # 准备数据并训练模型
    X, y = predictor.prepare_data()
    score = predictor.train_model(X, y)
    print(f'模型准确率: {score:.4f}')
    
    # 预测未来价格并绘图
    future_dates, predicted_prices = predictor.predict_future_trend()
    predictor.plot_results(future_dates, predicted_prices)
    print('预测图表已保存为 fuxingyiyao_trend_prediction_yuemo.png')
    
    # 输出当前价格
    current_price = predictor.df['Close'].iloc[-1]
    print(f'\n当前价格: {current_price:.2f}')
    
    # 输出每个月的预测价格
    print("\n每月预测价格:")
    for date, price in zip(future_dates, predicted_prices):
        print(f"{date.strftime('%Y-%m')}: {price:.2f}")
    
    # 计算���体涨跌幅
    total_change = ((predicted_prices[-1] - current_price) / current_price) * 100
    print(f"\n预测总体涨跌幅: {total_change:.2f}%")

if __name__ == '__main__':
    main()
