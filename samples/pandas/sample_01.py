# page 10
import pandas as pd

data = {'Name':     ["John",     "Anna",  "Peter",  "Linda"],
        'Location': ["New York", "Paris", "Nerlin", "London"],
        'Age':      [ 24,         13,      53,       33]
       }

data_pandas = pd.DataFrame(data)
print(data_pandas)
"""
   Age  Location   Name
0   24  New York   John
1   13     Paris   Anna
2   53    Nerlin  Peter
3   33    London  Linda
"""

print(data_pandas[data_pandas.Age > 30])
"""
   Age Location   Name
2   53   Nerlin  Peter
3   33   London  Linda

"""
