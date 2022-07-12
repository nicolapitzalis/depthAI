import csv
import iou

def sort_csv_folder(test_label, csv_files):
    
    csv_rows = []
    tl_rows = []
    new_csv_row = []
    
    included_cols = [0, 4, 5, 6, 7]
    iou_res = []

    for fcsv in csv_files:
        with open(fcsv, "r", encoding='UTF8') as f:
            reader = next(csv.reader(f))
            
            for row in reader:
                csv_rows.append(row)

        img_filename = fcsv.split('.csv')[0] + ".jpg"

        with open(test_label, "r", encoding='UTF8') as tl:
            reader = csv.reader(fcsv, delimiter=',')

            for row in reader:
                content = list(row[i] for i in included_cols)
            
                if content[0] in img_filename:
                    tl_rows.append(content)


        for crow in csv_rows:
            for tlrow in tl_rows:
                iou_res.append(iou.bb_intersection_over_union(crow[1:], tlrow[1:]))
            
            