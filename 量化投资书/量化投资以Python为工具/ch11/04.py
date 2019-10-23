import pandas as pd
import MySQLdb

mysql_cn = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='pwd123', db='stock')
df = pd.read_sql('select * from company limit 10;', con=mysql_cn)
mysql_cn.close()