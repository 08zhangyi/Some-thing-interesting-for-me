import pandas as pd
import candle

ssec2012 = pd.read_csv('ssec2012.csv')
ssec2012.index = ssec2012.iloc[:, 1]
ssec2012.index = pd.to_datetime(ssec2012.index, format="%Y-%m-%d")
ssec2012 = ssec2012.iloc[:, 2:]
Close = ssec2012.Close
Open = ssec2012.Open

ClOp = Close - Open
print(ClOp.describe())
Shape = [0, 0, 0]
lag1ClOp = ClOp.shift(1)
lag2ClOp = ClOp.shift(2)
for i in range(3, len(ClOp)):
    if all([lag2ClOp[i] < -11, abs(lag1ClOp[i])<2, ClOp[i]>6, abs(ClOp[i])>abs(lag2ClOp[i]*0.5)]):
        Shape.append(1)
    else:
        Shape.append(0)

lagOpen = Open.shift(1)
lagClose = Close.shift(1)
lag2Close = Close.shift(2)
Doji = [0, 0, 0]
for i in range(3, len(Open), 1):
    if all([lagOpen[i], lagOpen[i]<lag2Close[i], lagClose[i]<Open[i], (lagClose[i]<lag2Close[i])]):
        Doji.append(i)
    else:
        Doji.append(0)

ret = Close/Close.shift(1)-1
lag1ret = ret.shift(1)
lag2ret = ret.shift(2)
Trend = [0, 0, 0]
for i in range(3, len(ret)):
    if all([lag1ret[i]< lag2ret[i]<0]):
        Trend.append(1)
    else:
        Trend.append(0)

StarSig = []
for i in range(len(Trend)):
    if all([Shape[i]==1, Doji[i]==1, Trend[i]==1]):
        StarSig.append(1)
    else:
        StarSig.append(0)
for i in range(len(StarSig)):
    if StarSig[i] == 1:
        print(ssec2012.index[i])

ssec201209 = ssec2012['2012-08-21': '2012-09-30']
candle.candlePlot(ssec201209, title='上证综指2012年9月份的日K线图')
