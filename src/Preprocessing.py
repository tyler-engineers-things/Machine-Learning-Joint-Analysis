import torch
import csv
import sklearn


with open("Hexapod_One_Joint.csv", 'r', newline='') as csvfile:
    csv_dict_reader = csv.DictReader(csvfile)
    for row in csv_dict_reader:
        print(row['X'])
