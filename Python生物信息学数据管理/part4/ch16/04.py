from pylab import figure, title, xlabel, ylabel, hist, axis, grid, savefig

data = [1, 1, 9, 1, 3, 5, 8, 2, 1, 5, 11, 8, 3, 4, 2, 5]
n_bins = 5
figure()
num, bins, patches = hist(data, n_bins, normed=1.0, histtype='bar', facecolor='green', alpha=0.75)
title('Histogram')
xlabel('value')
ylabel('frequency')
axis()
grid(True)
savefig('histogram.png')