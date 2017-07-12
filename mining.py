import os
from bs4 import BeautifulSoup as parser

my_file = open("authors.csv", "a")
x = 0
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
                    print name
                    my_file.write(name)
                    my_file.write(",")
                    x+=1
                    break
                else:
                    print("removing1", fullname, "with author", name)
                    os.remove(fullname)
                    break
print "Total: ", x
