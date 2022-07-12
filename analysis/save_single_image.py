import cv2

img_path = "/Users/nicolapitzalis/Desktop/peopleDetector/test/c1s2_071096.jpg"

x = 1010
y = 399
w = 251
h = 699

image = cv2.imread(img_path)
image_to_save = image[y:y+h, x:x+w]
        
cv2.imwrite("/Users/nicolapitzalis/Desktop/peopleDetector/test_img.jpg", image_to_save)