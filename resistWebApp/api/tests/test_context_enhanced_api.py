import cv2
import os
import unittest
import sys
import numpy as np
import shutil

sys.path.insert(0, "../")


from context_enhanced_api import getRotatedImage, getSegmentationResult, deleteFiles, deleteFolder

# Test 1
class TestGetSegmentationResult(unittest.TestCase):

    def setUp(self):
        self.root = "../tests/images/"
        self.test_image = "test_segmentation.jpg"
        self.segmented_image = "segmented_image.png"

    def test_getSegmentationResult(self):

        # Call the function under test
        result = getSegmentationResult(self.test_image, self.root)

        segmented_image = cv2.imread(self.root + self.segmented_image, cv2.COLOR_BGR2GRAY)

        mask = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)
        mask[:, :, 1:] = 0
        mask[:, :, 2] = 1 * segmented_image
        alpha = np.where(segmented_image == 0, 0, 255).astype('uint8')
        mask = cv2.merge([mask, alpha])

        _, img_encoded = cv2.imencode('.png', mask)

        # Assert the result
        self.assertEqual(result.all(), img_encoded.all())

    def tearDown(self):
        # Clean up the test image and rotated image
        os.remove(self.root + self.segmented_image)


# Test 2
class TestGetRotatedImage(unittest.TestCase):

    def setUp(self):
        self.root = "../tests/images/"
        self.test_image = "test_rotation.jpg"
        self.rotated_image = "test_rotated_image.jpg"

    def test_getRotatedImage(self):

        # Call the function under test
        result = getRotatedImage(self.test_image, self.root)

        rotated_image = cv2.imread(self.root + self.rotated_image, cv2.COLOR_BGR2GRAY)
        _, img_encoded = cv2.imencode('.jpeg', rotated_image)

        # Assert the result
        self.assertEqual(result.all(), img_encoded.all())

    def tearDown(self):
        # Clean up the test image and rotated image
        os.remove(self.root + self.rotated_image)


# Test 3
class TestDeleteFiles(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and files for testing
        self.temp_dir = "./temp_dir"
        os.makedirs(self.temp_dir)
        self.file_list = [
            os.path.join(self.temp_dir, "file1.txt"),
            os.path.join(self.temp_dir, "file2.txt"),
            os.path.join(self.temp_dir, "file3.txt")
        ]
        for file in self.file_list:
            with open(file, "w"):
                pass

    def tearDown(self):
        # Remove the temporary directory and files after testing
        os.rmdir(self.temp_dir)

    def test_deleteFiles(self):
        # Call the deleteFiles function
        deleteFiles(self.file_list)

        # Assert that the files are deleted
        for file in self.file_list:
            self.assertFalse(os.path.exists(file))

# Test 4
class TestDeleteFolder(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and files for testing
        self.temp_dir = "./temp_dir"
        os.makedirs(self.temp_dir)

        # Create a subdirectory with files inside it
        sub_dir = os.path.join(self.temp_dir, "sub_dir")
        os.makedirs(sub_dir)
        with open(os.path.join(sub_dir, "file1.txt"), "w"):
            pass
        with open(os.path.join(sub_dir, "file2.txt"), "w"):
            pass

    def test_deleteFolder(self):
        # Call the deleteFolder function
        deleteFolder(self.temp_dir)

        # Assert that the folder no longer exists
        self.assertFalse(os.path.exists(self.temp_dir))

if __name__ == '__main__':
    unittest.main()