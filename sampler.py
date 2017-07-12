import os
import re
import numpy as np
from bs4 import BeautifulSoup as parser

my_file = open("training_data.csv", "a")
x = 0
filter_list = [" Unknown", " Anonymous", " Various", " Members of the Oxford Faculty of Modern History"]
for root, dirs, files in os.walk("./Data"):
    path = root.split(os.sep)
    for file in files:
        filename, file_extension = os.path.splitext(file)
        fullname = os.path.join(root, file)
        soup = parser(open(fullname).read(), "html.parser")
        started = False
        for str1 in soup.findAll('p', limit=100):
            if started and isinstance(str1.string, basestring):
                book += str1.string.replace("\n","")
            if len(str1.contents) > 0 and "Author: " in str1.contents[0]:
                name = str1.contents[0].split(": ")[1].replace(",","")
            if len(str1.contents) > 0 and "***START OF THE PROJECT GUTENBERG EBOOK" in str1.contents[0]:
                started = True
                book = ""
        #TODO: Change name to gender in dictionary located at gender_dictionary.npy
        read_dictionary = np.load('gender_dictionary.npy').item()
        if name in read_dictionary and ([name] != "m" or read_dictionary[name] != "f"):
            final_str = re.sub('[^A-Za-z]+', ' ', book) + "," + read_dictionary[name]
            #print (final_str)
            my_file.write(final_str)
            my_file.write("\n")
        else:
            print("Found null or error", name)
