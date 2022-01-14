import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gentxt():




    f = open("test_database.txt", "w")
    #f.write("LATITUDE LONGITUDE\n")
    with os.scandir('st_lucia/images/test/database') as folder:

        for image in folder:
            title = image.name
            name=title.split('@')
            f.write(name[5]+" "+name[6]+"\n")
    f.close()

    f = open("test_queries.txt", "w")
    #f.write("LATITUDE LONGITUDE\n")
    with os.scandir('st_lucia/images/test/queries') as folder:

        for image in folder:
            title = image.name
            name=title.split('@')
            f.write(name[5]+" "+name[6]+"\n")
    f.close()


if __name__ == "__main__":
    gentxt()
