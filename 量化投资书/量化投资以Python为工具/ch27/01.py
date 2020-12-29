import pandas as pd
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from datetime import datetime
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt

ssec2015 = pd.read_csv('SSEC2015.csv')
ssec2015 = ssec2015.iloc[:, 1:]
print(ssec2015.head(3))
print(ssec2015.iloc[-3:, :])

ssec2015.Date = [date2num(datetime.strptime(date, '%Y-%m-%d')) for date in ssec2015.Date]

print(type(ssec2015))

ssec2015list = list()
for i in range(len(ssec2015)):
    ssec2015list.append(ssec2015.iloc[i, :])

ax = plt.subplot()
mondays = WeekdayLocator(MONDAY)
weekFormatter = DateFormatter('%y %b %d')
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(DayLocator())
ax.xaxis.set_major_formatter(weekFormatter)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax.set_title("上证综指2015年3月份K线图")
candlestick_ohlc(ax, ssec2015list, width=0.7, colorup='r', colordown='g')
plt.setp(plt.gca().get_xticklabels(), rotation=50, horizontalalignment='center')
plt.show()

