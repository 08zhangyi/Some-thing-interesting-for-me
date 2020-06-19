import rpy2.robjects as robjects
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

r = robjects.r
pi = r.pi
x = r.c(1, 2, 3, 4, 5, 6)
y = r.seq(1, 10)
m = r.matrix(y, nrow=5)
n = r.matrix(y, ncol=5)
table = r("read.table('RandomDistribution.tsv',sep='\t')")
matrix = r.matrix(table, ncol=7)
mean_first_col = r.mean(matrix[0])
print(mean_first_col)

r = ro.r
table = r("read.table('Chi-square_input.txt', header=TRUE, sep='\t')")
print(r.names(table))
cont_table = r.table(table[1], table[2])
chitest = r['chisq.test']
print(chitest(table[1], table[2]))

r = ro.r
table = r("read.table('RandomDistribution.tsv',sep='\t')")
m = r.mean(table[2], trim=0, na_rm='FALSE')
sdev = r.sd(table[2], na_rm='FALSE')
value = 0.01844
zscore = (m[0] - value) / sdev[0]
print(zscore)
x = r.abs(zscore)
pvalue = r.pnorm(-x[0])
print(pvalue[0])

r = ro.r
r.plot(r.pnorm(100), xlab="y", ylab="y")

r = robjects.r
table = r("read.table('RandomDistribution.tsv',sep='\t')")
r.plot(table[1], table[2], xlab="x", ylab="y")
r.hist(table[4], xlab='x', main='Distribution of values')

r = ro.r
table = r("read.table('RandomDistribution.tsv',sep='\t')")
grdevices = importr('grDevices')
grdevices.png(file="Plot.png", width=512, height=512)
r.plot(table[1], table[2], xlab="x", ylab="y")
grdevices.dev_off()
grdevices.png(file="Histogram.png", width=512, height=512)
r.hist(table[4], xlab='x', main='Distribution of values')
grdevices.dev_off()