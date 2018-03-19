import pandas as pd

from sklearn.model_selection import train_test_split
import re


def CareerYear(x):
    if not x==x:
        return -1
    # 工作年限转换
    elif x.find("10+") > -1: #
        return 11
    elif x.find('< 1') > -1: # 将"< 1 year"转换成 0
        return 0
    else:
        return int(re.sub("\D", "", x))  # 其余数据，去掉“year”并转换成整数


folderOfData = ""
allData = pd.read_csv(folderOfData, header=0, encoding='latin1')
allData['term'] = allData['term'].apply(lambda x: int(x.replace(' months', '')))
# 标签：Fully Paid 正常用户；Charged Off 违约用户
allData['y'] = allData['loan_status'].map(lambda x: int(x == 'Charged Off'))
"""
贷款期限不同（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，
所以选取term==36个月的样本
"""
allData1 = allData.loc[allData.term == 36]
trainData, testData = train_test_split(allData1, test_size=0.4)

# 特征处理
trainData['int_rate_clean'] = trainData['int_rate'].map(lambda x: float(x.replace('%', ''))/100)
# 工作年限转化
trainData['emp_lenth_clean'] = trainData['emp_length'].map(CareerYear)










