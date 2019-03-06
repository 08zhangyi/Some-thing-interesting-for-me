import re

if re.search("\\\\", "I have one nee\dle") is not None:
    print("match it")
else:
    print("not match")

if re.search(r"\\", "I have one nee\dle") is not None:
    print("match it")
else:
    print("not match")