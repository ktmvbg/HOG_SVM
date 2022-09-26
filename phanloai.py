import os

lst = os.listdir("./output")

for file in lst:
    if("Circle" in file):
        os.rename("./output/"+file, "./Circle/"+file)
    elif("Triangle" in file):
        os.rename("./output/"+file, "./Triangle/"+file)
    elif("Square" in file):
        os.rename("./output/"+file, "./Square/"+file)
    else:
        os.remove("./output/"+file)