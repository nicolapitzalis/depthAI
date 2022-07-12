import cv2
import csv
import iou

def bounding_box_savings(output_dict, test_label, csv_folder, image_file, image_folder, detection_treshold=0.5, iou_treshold=0.5):

    width = 1920
    height = 1080
    flag = [-10]
    detection_boxes = output_dict['detection_boxes']
    detection_scores = output_dict['detection_scores']

    included_cols = [0, 4, 5, 6, 7, 8]
    header = ["id", "xmin", "ymin", "xmax", 'ymax']
    valid_boxes = []
    gt_boxes = []
    final_boxes = []

    # all_gt = glob.glob(os.path.join(gt_folder, "*.mat"))

    for i in range(output_dict['num_detections']):
        if(detection_scores[i] >= detection_treshold):
            valid_boxes.append(detection_boxes[i].tolist())

    len_valid_boxes = len(valid_boxes)

    image = cv2.imread(image_file)
    image_filename = (image_file.split('.jpg')[0]).split('/')[-1]

    # mat_file = gt_folder.join([f for f in all_gt if image_filename in f])
    # gt_boxes = sio.loadmat(mat_file)['box_new']

    with open(test_label, "r", encoding='UTF8') as tl:
            reader = csv.reader(tl, delimiter=',')

            for row in reader:
                content = list(row[i] for i in included_cols)
            
                if content[0] in image_filename + ".jpg":
                    gt_boxes.append([int(content[1]), int(content[2]), int(content[3]), int(content[4]), int(content[5])])

    len_gt_boxes = len(gt_boxes)
    
    final_boxes = [flag] * (max(len_valid_boxes, len_gt_boxes))
    
    csv_file = csv_folder + '/' + image_filename + '.csv'
    
    for i, box in enumerate(valid_boxes):
        y = int(height * box[0])
        x = int(width * box[1])
        w = int(width * box[3])     #xmax
        h = int(height * box[2])    #ymax
        
        iou_results = []
        pred = [x, y, w, h]
        
        # for box in gt_boxes:
        #     gts.append(box.tolist())

        #for every bounding box check the iou
        for gt in gt_boxes:
            x_gt = gt[1]
            y_gt = gt[2]
            w_gt = gt[3]
            h_gt = gt[4]
            gt_val = [x_gt, y_gt, w_gt, h_gt]
            iou_results.append(iou.bb_intersection_over_union(gt_val, pred))

        #take the greatest iou
        iou_final = max(iou_results)

        #take idx of the greatest iou to retrieve id
        idx = iou_results.index(iou_final)
        bb = gt_boxes[idx]

        if(iou_final < iou_treshold):
            bb[0] = -1
        
        row = [bb[0], x, y, w, h]
        final_boxes[idx] = row
        

    with open(csv_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        if flag in final_boxes:
            final_boxes = [e for e in final_boxes if e != flag]

        for j, row in enumerate(final_boxes):
            print(row)

            x = row[1]
            y = row[2]
            w = row[3]
            h = row[4]

            image_to_save = image[y:h, x:w]
            final_path = '{}/{}_{}.jpg'.format(image_folder, image_filename, j)
            cv2.imwrite(final_path, image_to_save)

            writer.writerow(row)

