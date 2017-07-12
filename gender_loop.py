import numpy as np
import os
from bs4 import BeautifulSoup as parser

dictionary = {}

#read_dictionary = np.load('gender_dictionary.npy').item()
#print(read_dictionary) # displays "world"

filter_list = [" Unknown", " Anonymous", " Various", " Members of the Oxford Faculty of Modern History"]
for root, dirs, files in os.walk("./Data"):
    path = root.split(os.sep)
    for file in files:
        filename, file_extension = os.path.splitext(file)
        fullname = os.path.join(root, file)
        soup = parser(open(fullname).read(), "html.parser")
        for str1 in soup.findAll('p', limit=20):
            if len(str1.contents) > 0 and "Author: " in str1.contents[0]:
                name = str1.contents[0].split(": ")[1].replace(",","")
                if not any(name in s for s in filter_list):
                    if not name in dictionary:
                        gender = raw_input(name + ": ")
                        dictionary[name] = gender
                    break
np.save('gender_dictionary.npy', dictionary)
