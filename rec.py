import os
import zipfile

for root, dirs, files in os.walk("."):
    path = root.split(os.sep)
    for file in files:
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".zip":
            print("unzipping ", file)
            zipfile.ZipFile(os.path.join(root, file)).extractall(os.path.join("."))
