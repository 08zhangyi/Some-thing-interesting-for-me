from operator import itemgetter

table = []
for line in open("random_distribution.tsv"):
    columns = line.split()
    columns = [float(x) for x in columns]
    table.append(columns)
column = 1
table_sorted = sorted(table, key = itemgetter(column))
for row in table_sorted:
    row = [str(x) for x in row]
    print("\t".join(row))

in_file = open("random_distribution.tsv")
table = []
for line in in_file:
    columns = line.split()
    columns = [float(x) for x in columns]
    table.append(columns)
table_sorted = sorted(table, key=itemgetter(0, 1, 2, 3, 4, 5, 6))
print(table_sorted)