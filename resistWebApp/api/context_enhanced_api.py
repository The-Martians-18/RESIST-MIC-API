# imports
import cv2
import numpy as np
import torch
from operator import itemgetter
import os
from shapely.geometry import Polygon
from scipy import ndimage as ndi
from skimage import morphology
from collections import defaultdict
import itertools
import math
import concurrent.futures
from gradio_client import Client
import shutil

## support functions to the support function
def getImageIndex(Array, imageID):
    imageIndex = -1
    for n in range(0, len(Array)):
        if Array[n][0] == imageID:
            imageIndex = n
            break

    return imageIndex

def getImageSize(imageList, imageID, patchSize):
    imageSize = [patchSize, patchSize]
    for image in imageList:
        if image[0] == imageID:
            imageSize = image[1]
            break
    return imageSize

def constructImageRow(rowPatches):
    sortedPatches = sorted(rowPatches, key=itemgetter(0))
    rowImge = torch.from_numpy(sortedPatches[0][1])
    for pt in sortedPatches[1:]:
        rowImge = torch.cat(((rowImge), torch.from_numpy(pt[1])), 1)
    return rowImge

## support functions

# function that breaks an image into patches
def createImagePatches(testImageSizes, testImages, patchSize, folderName):
    patchList = []
    patchHeight = patchSize
    patchWidth = patchSize

    for testImage in testImages:
        image_array = np.array(testImage)
        imageSize = image_array.shape
        imageHeight = imageSize[0]
        imageWidth = imageSize[1]
        testImageSizes.append(['testImage', [imageWidth, imageHeight]])

        for y in range(0, imageHeight, patchSize):
            for x in range(0, imageWidth, patchSize):
                if ((y + patchHeight > imageHeight) and (x + patchWidth > imageWidth)):
                    pat = testImage[y:imageHeight, x:imageWidth]
                    patch = np.pad(pat, ((0, y + patchHeight - imageHeight), (0, x + patchWidth - imageWidth), (0, 0)),
                                   'reflect')

                elif ((y + patchHeight <= imageHeight) and (x + patchWidth > imageWidth)):
                    pat = testImage[y:y + patchHeight, x:imageWidth]
                    patch = np.pad(pat, ((0, 0), (0, x + patchWidth - imageWidth), (0, 0)), 'reflect')

                elif ((y + patchHeight > imageHeight) and (x + patchWidth <= imageWidth)):
                    pat = testImage[y:imageHeight, x:x + patchWidth]
                    patch = np.pad(pat, ((0, y + patchHeight - imageHeight), (0, 0), (0, 0)), 'reflect')

                else:
                    patch = testImage[y:y + patchHeight, x:x + patchWidth]

                patchName = 'testImage' + "_" + str(int(y / patchSize)) + "_" + str(int(x / patchSize))
                patchPath = folderName + "/" + patchName + ".jpg"
                cv2.imwrite(patchPath, patch)
                patchList.append([patchPath, patchName])

    return patchList

# function that generate predictions for patches
def generatePatchPredictions(client, testImageSizes, patchList, patchSize):
    predictions = []
    for patch in patchList:
        mask = client.predict(
        				patch[0],	# str representing filepath or URL to image in 'image' Image component
        				api_name="/predict")
        print(mask)
        filesToDelete = []
        filesToDelete.append(mask)

        imageIndex = getImageIndex(predictions, patch[1][:9])

        imageSize = getImageSize(testImageSizes, patch[1][:9], patchSize)
        cordinates = (patch[1][10:]).split("_")
        predictionPatch = cv2.imread(mask)
        deleteFiles(filesToDelete)
        predictionPatch = cv2.cvtColor(predictionPatch, cv2.COLOR_BGR2GRAY)
        if imageIndex != -1:
            (predictions[imageIndex][1]).append([cordinates, predictionPatch])
        else:
            predictions.append(["testImage", [[cordinates, predictionPatch]], imageSize])
    return predictions

# function to reconstruct the patch predictions
def reconstructImage(ImagePatches, type):
    variable = True
    current_row = 0
    rows = []
    imageSize = ImagePatches[2]
    imageWidth = imageSize[0]
    imageHeight = imageSize[1]
    predictionPatches = ImagePatches[1]
    for n in range(0, len(predictionPatches)):
        y_cor = int(predictionPatches[n][0][0])
        x_cor = int(predictionPatches[n][0][1])

        if (current_row == y_cor and x_cor == 0):
            rowPatches = [[x_cor, predictionPatches[n][type]]]

        elif (current_row == y_cor and x_cor != 0):
            rowPatches.append([x_cor, predictionPatches[n][type]])

        else:
            numPatchesInRow = len(rowPatches) + 1
            newRow = constructImageRow(rowPatches)
            rows.append([current_row, newRow])
            rowPatches = [[x_cor, predictionPatches[n][type]]]
            current_row = y_cor
        if (n + 1 == len(predictionPatches)):
            newRow = constructImageRow(rowPatches)
            rows.append([y_cor, newRow])
            variable = False

    rows = sorted(rows, key=itemgetter(0))
    fullImage = rows[0][1]
    for row in rows[1:]:
        fullImage = torch.cat(((fullImage), (row[1])), 0)
    croppedImage = fullImage[0:imageHeight, 0:imageWidth]
    return croppedImage


# Apply context enhanced segmentation
def contextEnhancedFilter(noisyPredictions, noiseLessPredictions, patch_size, filterSize):
    filteredPredictions = []
    patchHeight = patch_size
    patchWidth = patch_size
    for num in range(0, len(noisyPredictions)):
        noisyImage = np.copy(noisyPredictions[num]) / 255
        noiseLessImage = np.copy(noiseLessPredictions[num] / 255)

        imageHeight, imageWidth = noisyPredictions[num].shape

        for y in range(0, imageHeight, filterSize):
            for x in range(0, imageWidth, filterSize):

                # Bottom - Right side corner of the image
                if ((y + patchHeight + filterSize > imageHeight) and (x + patchWidth + filterSize > imageWidth)):
                    filter = noiseLessImage[y - filterSize:imageHeight, x - filterSize:imageWidth]
                    num_of_zeros = np.count_nonzero(filter == 0)

                    if (num_of_zeros <= 0):
                        noisyImage[y:imageHeight, x:imageWidth] = 1

                # Top and Middle - Right side of the Image
                elif ((y + patchHeight + filterSize <= imageHeight) and (x + patchWidth + filterSize > imageWidth)):
                    if (y - filterSize >= 0):
                        filter = noiseLessImage[y - filterSize:y + patchHeight + filterSize, x - filterSize:imageWidth]
                    else:
                        filter = noiseLessImage[0:y + patchHeight + filterSize, x - filterSize:imageWidth]
                    num_of_zeros = np.count_nonzero(filter == 0)
                    if (num_of_zeros <= 5):
                        noisyImage[y:y + patchHeight, x:imageWidth] = 1

                # Bottom - Left and Middle parts of the Image
                elif ((y + patchHeight + filterSize > imageHeight) and (x + patchWidth + filterSize <= imageWidth)):
                    if (x - filterSize >= 0):
                        filter = noiseLessImage[y - filterSize:imageHeight, x - filterSize:x + patchWidth + filterSize]
                    else:
                        filter = noiseLessImage[y - filterSize:imageHeight, 0:x + patchWidth + filterSize]
                    num_of_zeros = np.count_nonzero(filter == 0)
                    if (num_of_zeros <= 5):
                        noisyImage[y:imageHeight, x:x + patchWidth] = 1

                else:
                    if ((x - filterSize >= 0) and (y - filterSize >= 0)):
                        filter = noiseLessImage[y - filterSize:y + patchHeight + filterSize,
                                 x - filterSize:x + patchWidth + filterSize]
                    elif ((x - filterSize >= 0) and (y - filterSize < 0)):
                        filter = noiseLessImage[0:y + patchHeight + filterSize,
                                 x - filterSize:x + patchWidth + filterSize]
                    elif ((x - filterSize < 0) and (y - filterSize < 0)):
                        filter = noiseLessImage[y - filterSize:y + patchHeight + filterSize,
                                 0:x + patchWidth + filterSize]
                    else:
                        filter = noiseLessImage[0:y + patchHeight + filterSize, 0:x + patchWidth + filterSize]
                    num_of_zeros = np.count_nonzero(filter == 0)
                    if (num_of_zeros <= 5):
                        noisyImage[y:y + patchHeight, x:x + patchWidth] = 1
        filteredPredictions.append(noisyImage)
    return filteredPredictions

# Drawing boxes for possible inverted channel contours
def getImageBoxes(filteredPredictions):
    image_boxes = []
    medianFilterPredictions = []

    for n in range(0, len(filteredPredictions)):
        boxes = []

        result = ndi.median_filter(filteredPredictions[n], size=1)

        result_c = np.copy(result)

        # medianFilterPredictions.append(result_c)
        result[result >= 0.5] = 255
        result[result < 0.5] = 0

        medianFilterPredictions.append(result)

        median_result = np.copy(result)

        median_result = np.pad(median_result, ((1, 1), (1, 1)), 'constant', constant_values=(255, 255))

        img_height, img_width = median_result.shape

        median_result = median_result.astype('uint8')
        ret, thresh = cv2.threshold(median_result, 0.5, 1, 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if len(contours[i]) >= 0:
                # rectangles
                rect = cv2.minAreaRect(contours[i])
                boxOriginal = cv2.boxPoints(rect)
                boxOriginal = np.int0(boxOriginal)
                (center, (w, h), angle) = rect  # take it apart
                if w < h:
                    distance = h
                else:
                    distance = w

                rectNew = (center, (w, h), angle)
                box = cv2.boxPoints(rectNew)
                box = np.int0(box)
                boxes.append([box, distance, contours[i], rect])

        image_boxes.append(boxes)

    return image_boxes, medianFilterPredictions

# Extending bounding boxes
def getExtendedBoxes(image_boxes):
    image_boxes_copies = image_boxes

    for img in range(0, len(image_boxes_copies)):
        imageE = sorted(image_boxes_copies[img], reverse=True, key=lambda x: x[1])
        whole_c = imageE.pop(0)
        (center_l, (w_l, h_l), angle_l) = imageE[0][3]
        distance_l = imageE[0][1]
        for boxIndex in range(0, len(imageE)):
            (center, (w, h), angle) = imageE[boxIndex][3]
            if w < h:
                if (h * h / w > distance_l):
                    h = (h * h / w) * (2 / 3)

                else:
                    h = h * h / w

            else:
                if (h > 0):
                    if (w * w / h > distance_l):
                        w = (w * w / h) * (2 / 3)

                    else:
                        w = w * w / h

                else:
                    w = w

            rectNew1 = (center, (w, h), angle)
            box = cv2.boxPoints(rectNew1)
            box = np.int0(box)
            imageE[boxIndex][0] = box
        image_boxes_copies[img] = imageE

    return image_boxes_copies

# Check if two extended boxes are overlapping
def does_overlap(rect1, rec2):
    polygon_1 = Polygon(rect1)
    polygon_2 = Polygon(rec2)

    intersect = polygon_1.intersection(
        polygon_2).area / polygon_1.union(polygon_2).area

    # Print the intersection percentage
    intersection = round(intersect * 100, 2)
    if (intersection > 0):
        return True
    else:
        return False

# Iteratively select the overlapping contours in the mask
def checkContourAcceptance(image_boxes_copies):
    copy_i_boxes = image_boxes_copies
    not_accepted_contours = []

    for image in copy_i_boxes:
        n_accepted = []
        imageE = sorted(image, reverse=True, key=lambda x: x[1])

        largest_c = imageE.pop(0)
        accepted_contours = [largest_c]
        c_detected = True
        while (c_detected):
            c_detected = False
            for a_element in accepted_contours:
                temp_accepted_indexes = []
                for r_element in range(0, len(imageE)):
                    overlap = does_overlap(a_element[0], imageE[r_element][0])
                    if (overlap == True):
                        c_detected = True
                        temp_accepted_indexes.append(r_element)

                sorted_indexes = sorted(temp_accepted_indexes, reverse=True)
                for s_index in sorted_indexes:
                    element = imageE.pop(s_index)
                    accepted_contours.append(element)

        not_accepted_contours.append(imageE)
    return not_accepted_contours

# Remove non overlapping bounding boxes
def applyBoundingBoxOverlap(medianFilterPredictions, not_accepted_contours, image_boxes_copies):
    prediction_copies = []
    for pred in medianFilterPredictions:
        new_pred = np.copy(pred)
        prediction_copies.append(new_pred)

    prediction_copies1 = []
    for pred in medianFilterPredictions:
        new_pred1 = np.copy(pred)
        prediction_copies1.append(new_pred1)

    copyiess = prediction_copies[:]
    copyiess1 = prediction_copies1[:]

    for image_n in range(0, len(copyiess)):
        not_accepted = not_accepted_contours[image_n]

        for not_acc in not_accepted:
            cv2.fillPoly(copyiess[image_n], [not_acc[2]], (1))

        for boxs in image_boxes_copies[image_n]:
            cv2.drawContours(copyiess1[image_n], [boxs[0]], 0, (0, 0, 255), 2)

        copyiess[image_n] = ndi.median_filter(copyiess[image_n], size=10)
        kernel_erode = np.ones((3, 3), np.uint8)
        copyiess[image_n] = cv2.erode(copyiess[image_n], kernel_erode, iterations=5)
        kernel_close = np.ones((9, 9), np.uint8)
        copyiess[image_n] = cv2.morphologyEx(copyiess[image_n], cv2.MORPH_OPEN, kernel_close)
        copyiess[image_n] = cv2.dilate(copyiess[image_n], kernel_erode, iterations=3)

    return copyiess

# Gap Filling & Interior Region Filling Functions
def skel(img):
  img = img.astype('uint8')
  inv_img = np.logical_not(img)
  threshold_value = 0.5
  image_binary = inv_img > threshold_value
  # inverted_image = np.logical_not(image_binary)
  bin_skeleton = morphology.skeletonize(image_binary)
  skeleton = bin_skeleton.astype(int)
  return skeleton

def getSkeltons(copyiess):
  skeletons = []
  for i in range(len(copyiess)):
    skeletons.append(skel(copyiess[i]))
  return skeletons

def getSkelCoords(img):
    img = img.astype('uint8')

    # Find row and column locations that are non-zero
    (rows, cols) = np.nonzero(img)

    # Initialize empty list of co-ordinates
    skel_coords = []

    # For each non-zero pixel...
    for (r, c) in zip(rows, cols):

        # Extract an 8-connected neighbourhood
        (col_neigh, row_neigh) = np.meshgrid(np.array([c - 1, c, c + 1]), np.array([r - 1, r, r + 1]))

        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')

        # Convert into a single 1D array and check for non-zero locations
        pix_neighbourhood = img[row_neigh, col_neigh].ravel() != 0

        # If the number of non-zero locations equals 2, add this to
        # our list of co-ordinates
        if np.sum(pix_neighbourhood) == 2:
            skel_coords.append((r, c))
    return skel_coords

def getClusterCoords(binary_image):
    # initialize set of input coordinates
    onePixels = set(zip(*np.where(binary_image == 1)))

    # initialize graph as a dictionary
    graph = defaultdict(list)

    # add edges between adjacent nodes
    for coord in onePixels:
        r, c = coord
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                neighbor = (r + dr, c + dc)
                if neighbor in onePixels:
                    graph[coord].append(neighbor)

    # initialize list of clusters and dictionary that maps coordinates to their cluster index
    clusters = []
    coord_to_cluster = {}

    # perform depth-first search to identify connected components
    visited = set()
    for start_node in onePixels:
        if start_node not in visited:
            cluster = []
            stack = [start_node]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    cluster.append(node)
                    coord_to_cluster[node] = len(clusters)
                    stack.extend(graph[node])
            clusters.append(cluster)

    return clusters, coord_to_cluster


def isConnected(point1, point2, connections):
    def helper(point, visited):
        visited.add(point)
        for connection in connections:
            if point in connection:
                other_point = connection[1] if connection[0] == point else connection[0]
                if other_point == point2 or (other_point not in visited and helper(other_point, visited)):
                    return True
        return False
    return helper(point1, set())

# Given lists
def getSkeletalPieceCoordinatesToConnect(lists):
    # Compute distances between each pair of lists
    distances = {}
    for list1, list2 in itertools.combinations(lists, 2):
        min_distance = math.inf
        min_coords = None
        for coord1 in list1:
            for coord2 in list2:
                distance = math.dist(coord1, coord2)
                if distance < min_distance:
                    min_distance = distance
                    min_coords = (coord1, coord2)
        distances[(tuple(list1), tuple(list2))] = (min_distance, min_coords)
    # Sort the coordinatesToConnect based on min_distance
    sorted_distances = sorted(distances.items(), key=lambda x: x[1][0])
    distances = dict(sorted_distances)
    coordinatesToConnect = []
    connections = []
    for pair, (distance, coords) in distances.items():
        list1, list2 = pair
        # if distance < 200 and (not isConnected(lists.index(list(list1)), lists.index(list(list2)), connections)):
        if distance < 250:
            connections.append([lists.index(list(list1)), lists.index(list(list2))])
            coordinatesToConnect.append(list(coords))
    return coordinatesToConnect


from skimage.draw import line


def set_line(coord1, coord2, image):
    rr, cc = line(coord1[0], coord1[1], coord2[0], coord2[1])
    image[rr, cc] = 1


from scipy import ndimage as ndi


def fillShapes(img):
    # Define the input binary image
    skeleton = np.copy(img)

    # Apply morphological operations to fill in interior pixels of connected components
    filled = ndi.binary_fill_holes(skeleton)
    return filled


def applyGapAndInteriorFilling(skeletons, copyiess):
    processed_results = []

    for i in range(len(skeletons)):
        img = np.copy(skeletons[i])
        skel_coords = getSkelCoords(img)
        clusters, coord_to_cluster = getClusterCoords(img)
        skeletalPieces = []
        for cluster in clusters:
            skeletalPieces.append([])
        for coord in skel_coords:
            skeletalPieces[coord_to_cluster[coord]].append(coord)

        coordinatesToConnect = getSkeletalPieceCoordinatesToConnect(skeletalPieces)
        zeros_array = np.zeros_like(img)
        for k in range(len(coordinatesToConnect)):
            set_line(coordinatesToConnect[k][0], coordinatesToConnect[k][1], img)
        for k in range(len(coordinatesToConnect)):
            set_line(coordinatesToConnect[k][0], coordinatesToConnect[k][1], zeros_array)

            # Define the structuring element for dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (71, 71))

        # Dilate the binary image to thicken the skeleton
        thickened = cv2.dilate(zeros_array.astype('uint8'), kernel)

        # Take the union of the two binary images
        union = np.logical_or(thickened, np.logical_not(copyiess[i])).astype(np.uint8)
        filled = fillShapes(img)
        union2 = np.logical_or(union, filled).astype(np.uint8)

        processed_results.append(np.logical_not(union2).astype(np.uint8))
    return processed_results

# Parallel Functions
def getContextDeprivedMask(image, folderNameDeprived, clientDeprived):
    print("deprived process started")
    patchSizeDeprived = 256
    print("patchSizeDeprived ", patchSizeDeprived)
    testImageSizesDeprived = []
    testImagesDeprived = [image]
    patchListDeprived = createImagePatches(testImageSizesDeprived, testImagesDeprived, patchSizeDeprived,
                                           folderNameDeprived)
    predictionsDeprived = generatePatchPredictions(clientDeprived, testImageSizesDeprived, patchListDeprived,
                                                   patchSizeDeprived)
    mask_dep = reconstructImage(predictionsDeprived[0], 1)
    mask_dep[mask_dep < 0.5] = 255
    mask_dep[mask_dep < 2] = 0
    # Get the image for validation in the test TestGetContextDeprivedMaskHelper
    # tensor_umat = mask_dep.numpy()
    # cv2.imwrite('test_cdm_helpered_image.png', tensor_umat)
    deleteFolder(folderNameDeprived)
    print("deprived process ended")
    return [mask_dep]


def getContextExtendedMask(image, folderNameExtended, clientExtended):
    print("extended process started")
    patchSizeExtended = 512
    print("patchSizeExtended ", patchSizeExtended)
    testImageSizesExtended = []
    testImagesExtended = [image]
    patchListExtended = createImagePatches(testImageSizesExtended, testImagesExtended, patchSizeExtended,
                                           folderNameExtended)
    predictionsExtended = generatePatchPredictions(clientExtended, testImageSizesExtended, patchListExtended,
                                                   patchSizeExtended)
    mask_exten = reconstructImage(predictionsExtended[0], 1)
    mask_exten[mask_exten < 0.5] = 255
    mask_exten[mask_exten < 2] = 0
    # Get the image for validation in the test TestGetContextExtendedMaskHelper
    # tensor_umat = mask_exten.numpy()
    # cv2.imwrite('mask_exten.png', tensor_umat)
    deleteFolder(folderNameExtended)
    print("extended process ended")
    return [mask_exten]

def getContextDeprivedMask_helper(imageName, root = "api/images/"):
    # call your original function here with arguments that can be pickled
    # imageName = "ESP_077800_1800.jpg"
    print(imageName, 'getContextDeprivedMask_helper')
    image = cv2.imread(root + imageName)

    folderNameDeprived = './Deprived-' + imageName[:-4]
    if not os.path.exists(folderNameDeprived):
        os.makedirs(folderNameDeprived)

    clientDeprived = Client("https://geethkavinda-resist-mic-context-deprived.hf.space/")
    # clientDeprived = Client("http://127.0.0.1:7860/")

    return getContextDeprivedMask(image, folderNameDeprived, clientDeprived)

def getContextExtendedMask_helper(imageName, root = "api/images/"):
    # call your original function here with arguments that can be pickled
    # imageName = "ESP_077800_1800.jpg"
    print(imageName, 'getContextExtendedMask_helper')
    image = cv2.imread(root + imageName)

    folderNameExtended = './Extended-' + imageName[:-4]
    if not os.path.exists(folderNameExtended):
        os.makedirs(folderNameExtended)

    clientExtended = Client("https://geethkavinda-resist-mic-context-extended.hf.space/")
    # clientExtended = Client("http://127.0.0.1:7861/")

    return getContextExtendedMask(image, folderNameExtended, clientExtended)

def makeMask(img):
    # Create a mask for the black pixels

    mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Set the green and blue channels to 0 for all pixels
    mask[:, :, 1:] = 0

    # Set the red channel to 255 for pixels with value 1 in img
    mask[:, :, 2] = 1 * img

    # Create an alpha channel where pixels with value 0 in img are fully transparent, and pixels with value 1 are fully opaque
    alpha = np.where(img == 0, 0, 255).astype('uint8')

    # Merge the alpha channel with the red mask to create a 4-channel image
    mask = cv2.merge([mask, alpha])

    # Save the image as a PNG with transparent background
    # cv2.imwrite("api/results/red_mask.png", mask)
    return mask

def deleteFolder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Loop through all the files and subdirectories in the folder
        for filename in os.listdir(folder_path):
            # Get the full file path
            file_path = os.path.join(folder_path, filename)
            # Check if it's a file or a subdirectory
            if os.path.isfile(file_path):
                # Use os.remove() to delete the file
                os.remove(file_path)
            else:
                # Use shutil.rmtree() to delete the subdirectory and all its contents
                shutil.rmtree(file_path)
        shutil.rmtree(folder_path)

def deleteFiles(fileList):
    for file in fileList:
        os.remove(file)


def getSegmentationResult(imageName, root = "api/images/"):
    originalImage = cv2.imread(root + imageName)
    gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    rotated_image = cv2.rotate(gray_image, cv2.ROTATE_180)
    histogramNormalizedImg = cv2.equalizeHist(rotated_image)
    newImgName = 'histogramNormalized_'+ imageName
    cv2.imwrite(root + newImgName, histogramNormalizedImg)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # submit the helper functions to the executor
        future1 = executor.submit(getContextDeprivedMask_helper, newImgName, root)
        future2 = executor.submit(getContextExtendedMask_helper, newImgName, root)

        # wait for the results to come back
        maskDeprived = future1.result()
        maskExtended = future2.result()

    filteredPredictions = contextEnhancedFilter(maskDeprived, maskExtended, 64, 64)

    image_boxes, medianFilterPredictions = getImageBoxes(filteredPredictions)

    image_boxes_copies = getExtendedBoxes(image_boxes)

    not_accepted_contours = checkContourAcceptance(image_boxes_copies)

    copyiess = applyBoundingBoxOverlap(medianFilterPredictions, not_accepted_contours, image_boxes_copies)

    skeletons = getSkeltons(copyiess)

    processed_results = applyGapAndInteriorFilling(skeletons, copyiess)

    pros_res = processed_results[0]
    pros_res[pros_res < 0.5] = 255
    pros_res[pros_res < 2] = 0

    # cv2.imwrite("api/results/processed_results1.png", pros_res)
    # k = cv2.imread('api/results/processed_results.png')

    # save the image for testing of segmentation
    # cv2.imwrite("segmented_image.png", pros_res)

    res = makeMask(pros_res)

    # cv2.imwrite("api/results/processed_results.png", res)
    deleteFiles([root + imageName, root + newImgName])

    _, img_encoded = cv2.imencode('.png', res)

    return img_encoded


def getRotatedImage(imageName, root = "api/images/"):
    originalImage = cv2.imread(root + imageName)
    gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    rotated_image = cv2.rotate(gray_image, cv2.ROTATE_180)
    _, img_encoded = cv2.imencode('.jpeg', rotated_image)
    deleteFiles([root + imageName])

    return img_encoded