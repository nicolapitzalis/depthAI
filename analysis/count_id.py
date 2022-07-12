import os
import glob
import csv

ROOT_DIR = os.path.relpath(os.path.join(os.path.dirname(__file__), '..'))
csv_path = os.path.join(ROOT_DIR, 'bb_dataset/csv_savings/')
csv_folder = glob.glob(os.path.join(csv_path, "*.csv"))

count = 0
count_bb = 0

for f in csv_folder:
    with open(f, "r", encoding="UTF8") as fcsv:
        csv_reader = csv.reader(fcsv, delimiter=',')
        next(csv_reader)

        for row in csv_reader:
            count_bb += 1
            if int(row[0]) > 0:
                count += 1

print(count)
print(count_bb)