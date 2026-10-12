import re

text = "Python is easy. Python is powerful. Python is popular."

sentences = re.split(r"(?<=[.!?])\s+", text)

print(sentences)
