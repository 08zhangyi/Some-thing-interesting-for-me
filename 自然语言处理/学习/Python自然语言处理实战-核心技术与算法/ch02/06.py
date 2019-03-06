import re

year_strings = []
strings = ['War of 1812', 'There are 5280 feet to a mile', 'Happy New Year 2016!']
for string in strings:
    if re.search('[1-2][0-9]{3}', string):
        year_strings.append(string)
print(year_strings)