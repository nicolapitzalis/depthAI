import cv2
import csv

def retrieve_gtbb(csv_file, img_path, save_path):
    
    #filename, xmin, ymin, xmax, ymax
    included_cols = [0, 5, 6, 7, 8]

    bb_for_img = []
    img_filename = img_path.split('.jpg')[0].split('/')[-1] + '.jpg'

    image = cv2.imread(img_path)

    with open(csv_file, "r", encoding='UTF8') as fcsv:
        csv_reader = csv.reader(fcsv, delimiter=',')

        for row in csv_reader:
            content = list(row[i] for i in included_cols)
            

            if content[0] in img_filename:
                bb_for_img.append(content)

        # print(bb_for_img)

    for i, box in enumerate(bb_for_img):
        x = int(box[1])
        y = int(box[2])
        w = int(box[3])
        h = int(box[4])
        
        image_to_save = image[y:h, x:w]
        
        final_path = '{}/{}_{}.jpg'.format(save_path, img_filename, i)
        cv2.imwrite(final_path, image_to_save)
