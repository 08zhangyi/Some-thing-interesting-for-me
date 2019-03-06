import re

years_string = '2016 was a good year, but 2017 will be better!'
years = re.findall('[2][0-9]{3}', years_string)
print(years)