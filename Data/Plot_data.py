import os
from glob import glob
import re
from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from txt_generator import gentxt



def range_coordinates():


    val = []
    max_g=np.array([0,-1000000])
    min_g=np.array([70,0])


    with os.scandir("./") as folder:
        for file in folder:
            if re.search(".txt", file.name):
                f = open(file.name, "r")
                tmp = np.loadtxt(f)
                max_g=np.vstack([max_g, tmp])
                max_g= max_g.max(0)
                min_g=np.vstack([min_g, tmp])
                min_g=min_g.min(0)
                f.close()

    print("max lat & long:", max_g, "\n", "min lat & long:", min_g)

def plt_data():

    data = {}
    with os.scandir("./") as folder:

        for file in folder:
            if re.search('.txt', file.name):
                f = open(file.name, "r")
                data[file.name] = np.loadtxt(f)
                f.close()

    coordinates = 0
    map_name = 0
    with os.scandir("./") as folder:

        for file in folder:
            if re.search("png", file.name):
                map_name = file.name
                tmp=map_name.strip('.png')
                coordinates = tmp.split('@')  # 1 to 4 are valuable
    print("map name:", map_name)
    print(coordinates)


    map = plt.imread("./"+map_name)
    plt.figure(1)
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(data['train_database.txt'][:, 1], data['train_database.txt'][:, 0],
               zorder=1, color='blue', label='database', alpha= 0.3, s=10)
    ax.scatter(data['train_queries.txt'][:, 1], data['train_queries.txt'][:, 0],
               zorder=2, color='red', label='queries', alpha= 0.1, s=10)
    ax.set_title('Plotting Train Database and Queries')

    ax.set_xlim(float(coordinates[1]), float(coordinates[2]))
    ax.set_ylim(float(coordinates[3]), float(coordinates[4]))
    ax.imshow(map, extent=[float(coordinates[1]), float(coordinates[2]),
              float(coordinates[3]), float(coordinates[4])], zorder=0)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend()

    plt.figure(2)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(data['test_database.txt'][:, 1], data['test_database.txt'][:, 0],
               zorder=1, color='blue', label='database', alpha=0.3, s=10)
    ax.scatter(data['test_queries.txt'][:, 1], data['test_queries.txt'][:, 0],
               zorder=2, color='red', label='queries', alpha=0.1, s=10)
    ax.set_title('Plotting Test Database and Queries')

    ax.set_xlim(float(coordinates[1]), float(coordinates[2]))
    ax.set_ylim(float(coordinates[3]), float(coordinates[4]))
    ax.imshow(map, extent=[float(coordinates[1]), float(coordinates[2]),
                           float(coordinates[3]), float(coordinates[4])], zorder=0)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend()

    plt.figure(3)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(data['val_database.txt'][:, 1], data['val_database.txt'][:, 0],
               zorder=1, color='blue', label='database', alpha=0.3, s=10)
    ax.scatter(data['val_queries.txt'][:, 1], data['val_queries.txt'][:, 0],
               zorder=2, color='red', label='queries', alpha=0.1, s=10)
    ax.set_title('Plotting Val Database and Queries')

    ax.set_xlim(float(coordinates[1]), float(coordinates[2]))
    ax.set_ylim(float(coordinates[3]), float(coordinates[4]))
    ax.imshow(map, extent=[float(coordinates[1]), float(coordinates[2]),
                           float(coordinates[3]), float(coordinates[4])], zorder=0)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #range_coordinates()
    plt_data()