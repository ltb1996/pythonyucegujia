import pandas as pd
import numpy as np
import pandas_ta as ta
def calculate_technical_indicators(csv_path):
    # 读取数据
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    # 重命名列
    column_mapping = {
        '日期': 'Date',
        '开盘': 'Open',
        '收盘': 'Close',
        '最高': 'High',
        '最低': 'Low',
        '成交量': 'Volume'
    }
    df = df.rename(columns=column_mapping)
    # 转换数值列
    for col in ['Open', 'Close', 'High', 'Low', 'Volume']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    # 计算技术指标
    # MA - 移动平均线 (5日，10日，20日)
    df['MA5'] = ta.sma(df['Close'], length=5)
    df['MA10'] = ta.sma(df['Close'], length=10)
    df['MA20'] = ta.sma(df['Close'], length=20)
    # RSI - 相对强弱指标 (14日)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    # MACD - 移动平均收敛散度
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    # CCI - 顺势指标 (14日)
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
    # ROC - 变动率 (10日)
    df['ROC'] = ta.roc(df['Close'], length=10)
    # AO - Awesome Oscillator
    df['AO'] = ta.ao(df['High'], df['Low'])
    # ADX - 平均趋向指标 (14日)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    # TR - 真实波幅
    df['TR'] = ta.true_range(df['High'], df['Low'], df['Close'])
    # 将所有技术指标四舍五入到两位小数
    round_columns = ['MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 
                    'MACD_Hist', 'CCI', 'ROC', 'AO', 'ADX', 'TR']
    for col in round_columns:
        df[col] = df[col].round(2)
    # 将日期设为索引
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # 删除NaN值
    df = df.dropna()
    # 保存结果到新的CSV文件
    output_path = 'xiaomi_technical_indicators.csv'
    df.to_csv(output_path, float_format='%.2f')  # 使用 float_format 确保CSV文件中也是两位小数
    print(f"技术指标已计算完成并保存至 {output_path}")
    # 返回最近的5行数据作为示例
    return df.tail()
# 使用函数
csv_path = r'd:\benkeshengxm\pythonyucegujia\xiaomiproject\01810.csv'  # 使用原始字符串
# 或者
# csv_path = 'd:\\benkeshengxm\\pythonyucegujia\\xiaomiproject\\01810.csv'  # 使用双反斜杠
recent_data = calculate_technical_indicators(csv_path)
print("\n最近5天的技术指标数据：")
print(recent_data)