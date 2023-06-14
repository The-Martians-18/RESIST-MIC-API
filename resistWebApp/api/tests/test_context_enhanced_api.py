import cv2
import os
import unittest
import sys
import numpy as np

sys.path.insert(0, "../")


from context_enhanced_api import getRotatedImage, getSegmentationResult

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


if __name__ == '__main__':
    unittest.main()
