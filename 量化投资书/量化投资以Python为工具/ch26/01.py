import pandas as pd
import numpy as np
from arch.unitroot import ADF
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re

sh = pd.read_csv('sh50p.csv', index_col='Trddt')
sh.index = pd.to_datetime(sh.index)
formStart = '2014-01-01'
formEnd = '2015-01-01'
shform = sh[formStart:formEnd]
print(shform.head(2))
PAf = shform['601988']
PBf = shform['600000']
pairf = pd.concat([PAf, PBf], axis=1)
print(len(pairf))


def SSD(priceX, priceY):
    if priceX is None or priceY is None:
        print('缺少价格序列.')
    returnX = (priceX - priceX.shift(1)) / priceX.shift(1)[1:]
    returnY = (priceY - priceY.shift(1)) / priceY.shift(1)[1:]
    standardX = (returnX + 1).cumprod()
    standardY = (returnY + 1).cumprod()
    SSD = np.sum((standardX - standardY)**2)
    return SSD


dis = SSD(PAf, PBf)
print(dis)

PAflog = np.log(PAf)
adfA = ADF(PAflog)
print(adfA.summary().as_text())
retA = PAflog.diff()[1:]
adfretA = ADF(retA)
print(adfretA.summary().as_text())

PBflog = np.log(PBf)
adfB = ADF(PBflog)
print(adfB.summary().as_text())
retB = PBflog.diff()[1:]
adfretB = ADF(retB)
print(adfretB.summary().as_text())

PAflog.plot(label='601988', style='--')
PBflog.plot(label='600000', style='-')
plt.legend(loc='upper left')
plt.title('中国银行与浦发银行的对数价格时序图')

retA.plot(label='601988')
retB.plot(label='600000')
plt.legend(loc='lower left')
plt.title('中国银行与浦发银行的对数价差分（收益率）')

model = sm.OLS(PBflog, sm.add_constant(PAflog))
results = model.fit()
print(results.summary())

alpha = results.params[0]
beta = results.params[1]
spread = PBflog - beta*PAflog - alpha
print(spread.head())
spread.plot()
plt.title('价差序列')
adfSpread = ADF(spread, trend='c')
print(adfSpread.summary().as_text())

standardA = (1+retA).cumprod()
standardB = (1+retB).cumprod()
SSD_pair = standardB - standardA
print(SSD_pair.head())
meanSSD_pair = np.mean(SSD_pair)
sdSSD_pair = np.std(SSD_pair)
thresholdUP = meanSSD_pair + 1.2*sdSSD_pair
thresholdDown = meanSSD_pair - 1.2*sdSSD_pair
SSD_pair.plot()
plt.title('中国银行与浦发银行标准化价差序列（形成期)')
plt.axhline(y=meanSSD_pair, color='black')
plt.axhline(y=thresholdUP, color='green')
plt.axhline(y=thresholdDown, color='green')

tradStart = '2015-01-01'
tradEnd = '2015-06-30'
PAt = sh.loc[tradStart:tradEnd, '601988']
PBt = sh.loc[tradStart:tradEnd, '600000']


def spreadCal(x, y):
    retx = (x - x.shift(1)) / x.shift(1)[1:]
    rety = (y - y.shift(1)) / y.shift(1)[1:]
    standardX = (1+retx).cumprod()
    standardY = (1+rety).cumprod()
    spread = standardX - standardY
    return spread


TradSpread = spreadCal(PBt, PAt).dropna()
print(TradSpread.describe())
TradSpread.plot()
plt.title('交易期价差序列')
plt.axhline(y=meanSSD_pair, color='black')
plt.axhline(y=thresholdUP, color='green')
plt.axhline(y=thresholdDown, color='green')

spreadf = PBflog - beta * PAflog - alpha
mu = np.mean(spreadf)
sd = np.std(spreadf)
CoSpreadT = np.log(PBt) - beta * np.log(PAt) - alpha
print(CoSpreadT.describe())
CoSpreadT.plot()
plt.title('交易期价差序列（协整配对）')
plt.axhline(y=mu, color='black')
plt.axhline(y=mu+1.2*sd, color='green')
plt.axhline(y=mu-1.2&sd, color='green')


class PairTrading:
    def SSD(self, priceX, priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列')
        returnX = (priceX - priceX.shift(1)) / priceX.shift(1)[1:]
        returnY = (priceY - priceY.shift(1)) / priceY.shift(1)[1:]
        standardX = (returnX + 1).cumprod()
        standardY = (returnY + 1).cumprod()
        SSD = np.sum((standardY - standardX)**2)
        return SSD

    def SSDSpread(self, priceX, priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列')
        retx = (priceX - priceX.shift(1)) / priceX.shift(1)[1:]
        rety = (priceY - priceY.shift(1)) / priceY.shift(1)[1:]
        standardX = (retx + 1).cumprod()
        standardY = (rety + 1).cumprod()
        spread = standardY - standardX
        return spread

    def cointegration(self, priceX, priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列')
        priceX = np.log(priceX)
        priceY = np.log(priceY)
        results = sm.OLS(priceY, sm.add_constant(priceX)).fit()
        resid = results.resid
        adfSpread = ADF(resid)
        if adfSpread.pvalue >= 0.05:
            print('''交易价格不具有协整关系。
            P-value of ADF test: %f
            Coefficients of regression:
            Intercept: %f
            Beta: %f
            ''' % (adfSpread.pvalue, results.params[0], results.params[1]))
            return None
        else:
            print('''交易价格具有协整关系。
            P-value of ADF test: %f
            Coefficients of regression:
            Intercept: %f
            Beta: %f
            ''' % (adfSpread.pvalue, results.params[0], results.params[1]))
            return results.params[0], results.params[1]

    def CointegrationSpread(self, priceX, priceY, formPeriod, tradePeriod):
        if priceX is None or priceY is None:
            print('缺少价格序列')
        if not (re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', formPeriod)
        or re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', tradePeriod)):
            print('形成期或交易期格式错误。')
        formX = priceX[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
        formY = priceY[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
        coefficients = self.cointegration(formX, formY)
        if coefficients is None:
            print('未形成协整关系，无法配对。')
        else:
            spread = (np.log(priceY[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]])
                      -coefficients[0]-coefficients[1]*np.log(priceX[tradePeriod.split(':')[0]:
                                                              tradePeriod.split(':')[1]]))
            return spread

    def calBound(self, priceX, priceY, method, formPeriod, width=1.5):
        if not re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', formPeriod):
            print('形成期格式错误。')
        if method == 'SSD':
            spread = self.SSDSpread(priceX[formPeriod.split(':')[0]:formPeriod.split(':')[1]],
                                    priceY[formPeriod.split(':')[0]:formPeriod.split(':')[1]])
            mu = np.mean(spread)
            sd = np.std(spread)
            UpperBound = mu + width * sd
            LowerBound = mu - width * sd
            return UpperBound, LowerBound
        elif method == 'Cointegration':
            spread = self.CointegrationSpread(priceX, priceY, formPeriod, formPeriod)
            mu = np.mean(spread)
            sd = np.std(spread)
            UpperBound = mu + width * sd
            LowerBound = mu - width * sd
            return UpperBound, LowerBound
        else:
            print('不存在该方法，请选择"SSD"或是"Cointegration"。')


formPeriod = '2014-01-01:2015-01-01'
tradePeriod = '2015-01-01:2015-06-30'
priceA = sh['601988']
priceB = sh['600000']
priceAf = priceA[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
priceBf = priceB[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
priceAt = priceA[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]]
priceBt = priceB[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]]
pt = PairTrading()
SSD = pt.SSD(priceAf, priceBf)
print(SSD)
SSDspread = pt.SSDSpread(priceAf, priceBf)
print(SSDspread.describe())
print(SSDspread.head())
coefficients = pt.cointegration(priceAf, priceBf)
print(coefficients)
CoSpreadF = pt.CointegrationSpread(priceA, priceB, formPeriod, formPeriod)
print(CoSpreadF.head())
CoSpreadTr = pt.CointegrationSpread(priceA, priceB, formPeriod, tradePeriod)
print(CoSpreadTr.describe())
bound = pt.calBound(priceA, priceB, 'Cointegration', formPeriod, width=1.2)
print(bound)

formStart = '2014-01-01'
formEnd = '2015-01-01'
PA = sh['601988']
PB = sh['600000']
PAf = PA[formStart:formEnd]
PBf = PB[formStart:formEnd]
log_PAf = np.log(PAf)
adfA = ADF(log_PAf)
print(adfA.summary().as_text())
log_PBf = np.log(PBf)
adfB = ADF(log_PBf)
print(adfB.summary().as_text())
adfBd = ADF(log_PBf.diff()[1:])
print(adfBd.summary().as_text())
model = sm.OLS(log_PBf, sm.add_constant(log_PAf)).fit()
print(model.summary())
alpha = model.params[0]
print(alpha)
beta = model.params[1]
print(beta)
spreadf = log_PBf - beta * log_PAf - alpha
adfSpread = ADF(spreadf)
print(adfSpread.summary().as_text())

mu = np.mean(spreadf)
sd = np.std(spreadf)

tradeStart = '2015-01-01'
tradeEnd = '2015-06-30'
PAt = PA[tradeStart:tradeEnd]
PBt = PB[tradeStart:tradeEnd]
CoSpreadT = np.log(PBt) - beta * np.log(PAt) - alpha
print(CoSpreadT.describe())
CoSpreadT.plot()
plt.title('交易期价差序列（协整配对）')
plt.axhline(y=mu, color='black')
plt.axhline(y=mu+0.2*sd, color='blue', ls='-', lw=2)
plt.axhline(y=mu-0.2*sd, color='blue', ls='-', lw=2)
plt.axhline(y=mu+1.5*sd, color='green', ls='-', lw=2.5)
plt.axhline(y=mu-1.5*sd, color='green', ls='-', lw=2.5)
plt.axhline(y=mu+2.5*sd, color='red', ls='-', lw=3)

level = (float('-inf'), mu-2.5*sd, mu-1.5*sd, mu-0.2*sd, mu+0.2*sd, mu+1.5*sd, mu-1.5*sd, float('-inf'))
prcLevel = pd.cut(CoSpreadT, level, labels=False)-3
print(prcLevel.head())


def TradeSig(prcLevel):
    n = len(prcLevel)
    signal = np.zeros(n)
    for i in range(1, n):
        if prcLevel[i-1]==1 and prcLevel[i]==2:
            signal[i]=-2
        elif prcLevel[i-1]==1 and prcLevel[i]==0:
            signal[i]=2
        elif prcLevel[i-1]==2 and prcLevel[i]==3:
            signal[i]=3
        elif prcLevel[i-1]==-1 and prcLevel[i]==-2:
            signal[i]=1
        elif prcLevel[i-1]==-1 and prcLevel[i]==0:
            signal[i]=-1
        elif prcLevel[i-1]==-2 and prcLevel[i]==-3:
            signal[i]=-3
    return signal


signal = TradeSig(prcLevel)
position = [signal[0]]
ns = len(signal)
for i in range(1, ns):
    position.append(position[-1])
    if signal[i]==1:
        position[i]=1
    elif signal[i]==-2:
        position[i]=-1
    elif signal[i]==-1 and position[i-1]==1:
        position[i]=0
    elif signal[i]==2 and position[i-1]==-1:
        position[i]=0
    elif signal[i]==3:
        position[i]=0
    elif signal[i]==-3:
        position[i]=0
    position = pd.Series(position, index=CoSpreadT.index)
    print(position.tail())


def TradeSim(priceX, priceY, position):
    n = len(position)
    size = 1000
    shareY = size * position
    shareX = [(-beta)*shareY[0]*priceY[0]/priceX[0]]
    cash = [2000]
    for i in range(1, n):
        shareX.append(shareX[i-1])
        cash.append(cash[i-1])
        if position[i-1]==0 and position[i]==1:
            shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
            cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
        elif position[i-1]==0 and position[i]==-1:
            shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
            cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
        elif position[i-1]==1 and position[i]==0:
            shareX[i] = 0
            cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])
        elif position[i-1]==-1 and position[i]==0:
            shareX[i] = 0
            cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])
    cash = pd.Series(cash, index=position.index)
    shareX = pd.Series(shareX, index=position.index)
    shareY = pd.Series(shareY, index=position.index)
    asset = cash + shareY * priceY + shareX * priceX
    account = pd.DataFrame({'Position': position, 'ShareY': shareY, 'ShareX': shareX, 'Cash': cash, 'Asset': asset})
    return account


account = TradeSim(PAt, PBt, position)
print(account.tail())
account.iloc[:, [0, 1, 4]].plot(style=['--', '-', ':'])