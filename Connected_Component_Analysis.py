import matplotlib.pyplot as plt
import numpy as np
import cv2

def preprocessingOfImages(img, img_name):

    global finalProcessedImage
    if img_name == 'birds1.jpg':
        thresholdImage = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)[1]
        erodedImage = cv2.erode(thresholdImage, np.ones((3, 3), np.uint8), iterations=1)
        finalProcessedImage = cv2.dilate(erodedImage, np.ones((3, 3), np.uint8), iterations=1)

    if img_name == 'birds2.jpg':
        thresholdImage = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)[1]
        erodedImage = cv2.erode(thresholdImage, np.ones((3, 3), np.uint8), iterations=1)
        finalProcessedImage = cv2.dilate(erodedImage, np.ones((3, 3), np.uint8), iterations=1)

    if img_name == 'birds3.jpg':
        thresholdImage = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)[1]
        erodedImage = cv2.erode(thresholdImage, np.ones((2, 2), np.uint8), iterations=1)
        finalProcessedImage = cv2.dilate(erodedImage, np.ones((6, 6), np.uint8), iterations=1)

    if img_name == 'demo4.png':
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        grayImage = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        filteredImage1 = cv2.medianBlur(grayImage,5)
        filteredImage2 = cv2.medianBlur(filteredImage1, 5)
        thresholdImage = cv2.threshold(filteredImage2, 110, 255, cv2.THRESH_BINARY)[1]
        erodedImage = cv2.erode(thresholdImage, np.ones((2, 2), np.uint8), iterations=1)
        finalProcessedImage = cv2.dilate(erodedImage, np.ones((4, 4), np.uint8), iterations=1)

    if img_name == 'dice5.PNG':
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        grayImage = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        thresholdImage = cv2.threshold(grayImage, 150, 255, cv2.THRESH_BINARY_INV)[1]
        dilatedImage = cv2.dilate(thresholdImage, np.ones((3, 3), np.uint8), iterations=1)
        erosedImage = cv2.erode(dilatedImage, np.ones((4, 4), np.uint8), iterations=1)
        erosedImage2 = cv2.erode(erosedImage, np.ones((2, 2), np.uint8), iterations=1)
        finalProcessedImage = cv2.dilate(erosedImage2, np.ones((6, 6), np.uint8), iterations=1)

    if img_name == 'dice6.PNG':
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        grayImage = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        thresholdImage = cv2.threshold(grayImage, 150, 255, cv2.THRESH_BINARY_INV)[1]
        dilatedImage = cv2.dilate(thresholdImage, np.ones((2, 2), np.uint8), iterations=1)
        erosedImage = cv2.erode(dilatedImage, np.ones((3, 3), np.uint8), iterations=1)
        erosedImage2 = cv2.erode(erosedImage, np.ones((2, 2), np.uint8), iterations=1)
        finalProcessedImage = cv2.dilate(erosedImage2, np.ones((2, 2), np.uint8), iterations=1)

    plt.imshow(finalProcessedImage, cmap="gray")
    plt.title('dice6')
    plt.show()

    return finalProcessedImage

if __name__ == "__main__":

    img = "dice6.PNG"
    img_name = "dice6.PNG"

    image_1 = cv2.imread(img, 0)
    dimensions = np.shape(image_1)
    width = image_1.shape[0]
    height = image_1.shape[1]
    image_2 = np.zeros(dimensions, dtype=np.uint16)

    image_1 = preprocessingOfImages(image_1, img_name)

    equivalency_list = [0]
    Label = 0
    for row in range(0, dimensions[0]):
        for col in range(0, dimensions[1]):
            if image_1[row, col] == 255:
                top_value = image_2[row - 1, col]
                backward_value = image_2[row, col - 1]
                if top_value == 0 and backward_value == 0:
                    Label = Label + 1
                    equivalency_list.append(Label)
                    image_2[row, col] = Label
                elif top_value == 0 and backward_value != 0:
                    image_2[row, col] = backward_value
                elif top_value != 0 and backward_value == 0:
                    image_2[row, col] = top_value
                elif top_value == backward_value:
                    image_2[row, col] = top_value
                else:
                    a = min(equivalency_list[top_value], equivalency_list[backward_value])
                    b = max(equivalency_list[top_value], equivalency_list[backward_value])
                    c = equivalency_list[b]
                    image_2[row, col] = a
                    for p in range(np.shape(equivalency_list)[0]):
                        if equivalency_list[p] == b:
                            equivalency_list[p] = a

    print("No of Objects in image are:", np.count_nonzero(np.unique(equivalency_list)))