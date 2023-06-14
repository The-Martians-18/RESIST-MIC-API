from django.test import TestCase
from django.urls import reverse
from django.http import HttpResponse
from rest_framework import status
from rest_framework.test import APIClient
from unittest.mock import patch
from . import imageHelpers
from . import context_enhanced_api
import json
import cv2
import os
import unittest
import sys
import numpy as np
import shutil

sys.path.insert(0, "../")


class ImageViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch('api.views.imageHelpers.getImageDetails')
    def test_get_image_details(self, mock_get_image_details):
        valid_image_id = "ESP_072116_1740"
        invalid_image_id = "ESP_072116_1940"

        valid_expected_response = {
            "title": "Sinuous Ridge in Aeolis Dorsa",
            "image_id": "ESP_072116_1740",
            "latitude": "-5.731°",
            "longitude": "152.681°"
        }

        invalid_expected_response = {
            "error": "Invalid image ID."
        }

        # Mock the getImageDetails function to return the expected responses
        mock_get_image_details.side_effect = [valid_expected_response, invalid_expected_response]

        # Test for valid image ID
        valid_url = reverse('get_image_details', args=[valid_image_id])
        valid_response = self.client.get(valid_url)

        self.assertEqual(valid_response.status_code, status.HTTP_200_OK)
        self.assertEqual(valid_response.data['title'], valid_expected_response['title'])
        self.assertEqual(valid_response.data['image_id'], valid_expected_response['image_id'])
        self.assertEqual(valid_response.data['latitude'], valid_expected_response['latitude'])
        self.assertEqual(valid_response.data['longitude'], valid_expected_response['longitude'])

        # Test for invalid image ID
        invalid_url = reverse('get_image_details', args=[invalid_image_id])
        invalid_response = self.client.get(invalid_url)

        self.assertEqual(invalid_response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(invalid_response.data['error'], invalid_expected_response['error'])

    @patch('api.views.getImage')
    def test_get_image(self, mock_get_image):
        valid_image_id = "ESP_072116_1740"
        invalid_image_id = "ESP_072116_1940"

        valid_expected_response = HttpResponse(b'', content_type='image/jpeg')

        invalid_expected_response = {
            "error": "Image not found. Invalid image ID."
        }

        # Mock the getImageDetails function to return the expected responses
        mock_get_image.side_effect = [valid_expected_response, invalid_expected_response]

        # Test for valid image ID
        valid_url = reverse('get_image', args=[valid_image_id])
        valid_response = self.client.get(valid_url)

        self.assertEqual(valid_response.status_code, status.HTTP_200_OK)
        self.assertEqual(valid_response['Content-Type'], valid_expected_response['Content-Type'])
        # self.assertEqual(len(valid_response.content), len(valid_expected_response.content))

        # Test for invalid image ID
        invalid_url = reverse('get_image', args=[invalid_image_id])
        invalid_response = self.client.get(invalid_url)

        self.assertEqual(invalid_response.status_code, status.HTTP_404_NOT_FOUND)
        content = json.loads(invalid_response.content)
        self.assertEqual(content['error'], invalid_expected_response['error'])

    @patch('api.views.getImageDetails')
    def test_get_image_segmentation(self, mock_get_image_segmentation):
        valid_image_id = "ESP_072116_1740"
        invalid_image_id = "ESP_072116_1940"

        valid_expected_response = HttpResponse(b'', content_type='image/jpeg')

        invalid_expected_response = {
            "error": "Image not found. Invalid image ID."
        }

        # Mock the getImageDetails function to return the expected responses
        mock_get_image_segmentation.side_effect = [valid_expected_response, invalid_expected_response]

        # Test for valid image ID

        # Test for invalid image ID
        invalid_url = reverse('get_image_segmentation', args=[invalid_image_id])
        invalid_response = self.client.get(invalid_url)

        self.assertEqual(invalid_response.status_code, status.HTTP_404_NOT_FOUND)
        content = json.loads(invalid_response.content)
        self.assertEqual(content['error'], invalid_expected_response['error'])
    
    @patch('api.views.imageHelpers.scrapeImageData')
    def test_get_images_in_boundary(self, mock_scrape_image_data):
        # Define the expected response
        expected_response = {
            "images": [
                {
                    "title": "Layering in an Exhumed Crater at Meridiani Planum",
                    "image_id": "PSP_001374_1805",
                    "latitude": "0.664°",
                    "longitude": "7.396°",
                    "thumbnailLink": "https://static.uahirise.org/images/2010/thumb/PSP_001374_1805.jpg"
                },
                {
                    "title": "Meridiani Planum\r",
                    "image_id": "ESP_045311_1810",
                    "latitude": "0.856°",
                    "longitude": "7.181°",
                    "thumbnailLink": "https://static.uahirise.org/images/2016/thumb/ESP_045311_1810.jpg"
                },
                {
                    "title": "Monitoring Change in Meridiani Planum\r",
                    "image_id": "ESP_028630_1805",
                    "latitude": "0.677°",
                    "longitude": "7.401°",
                    "thumbnailLink": "https://static.uahirise.org/images/2012/thumb/ESP_028630_1805.jpg"
                },
                {
                    "title": "Contrasting Thermal Inertia Units in Meridiani Planum\r",
                    "image_id": "ESP_047869_1810",
                    "latitude": "0.882°",
                    "longitude": "7.687°",
                    "thumbnailLink": "https://static.uahirise.org/images/2016/thumb/ESP_047869_1810.jpg"
                },
                {
                    "title": "Monitoring Change in Meridiani Planum\r",
                    "image_id": "ESP_028564_1805",
                    "latitude": "0.672°",
                    "longitude": "7.399°",
                    "thumbnailLink": "https://static.uahirise.org/images/2012/thumb/ESP_028564_1805.jpg"
                }
            ]
        }

        # Mock the scrapeImageData function to return the expected response
        mock_scrape_image_data.return_value = expected_response["images"]

        # Prepare the request data
        request_data = {
            "latitude_beginning": 0,
            "latitude_ending": 1,
            "longitude_beginning": 7,
            "longitude_ending": 8
        }

        # Send a POST request to the endpoint
        url = reverse('get_images_in_boundary')
        response = self.client.post(url, data=request_data)

        # Assert the response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, expected_response)



class ImageHelpersTestCase(TestCase):

    json_data = {
            "latitude_beginning": 0,
            "latitude_ending": 1,
            "longitude_beginning": 7,
            "longitude_ending": 8
        }

    expected_images = [
            {
                'title': 'Monitoring Change in Meridiani Planum\r',
                'image_id': 'ESP_028564_1805',
                'latitude': '0.672°',
                'longitude': '7.399°',
                'thumbnailLink': 'https://static.uahirise.org/images/2012/thumb/ESP_028564_1805.jpg'
            },
            {
                'title': 'Contrasting Thermal Inertia Units in Meridiani Planum\r',
                'image_id': 'ESP_047869_1810',
                'latitude': '0.882°',
                'longitude': '7.687°',
                'thumbnailLink': 'https://static.uahirise.org/images/2016/thumb/ESP_047869_1810.jpg'
            },
            {
                'title': 'Meridiani Planum\r',
                'image_id': 'ESP_045311_1810',
                'latitude': '0.856°',
                'longitude': '7.181°',
                'thumbnailLink': 'https://static.uahirise.org/images/2016/thumb/ESP_045311_1810.jpg'
            },
            {
                'title': 'Monitoring Change in Meridiani Planum\r',
                'image_id': 'ESP_028630_1805',
                'latitude': '0.677°',
                'longitude': '7.401°',
                'thumbnailLink': 'https://static.uahirise.org/images/2012/thumb/ESP_028630_1805.jpg'
            },
            {
                'title': 'Layering in an Exhumed Crater at Meridiani Planum',
                'image_id': 'PSP_001374_1805',
                'latitude': '0.664°',
                'longitude': '7.396°',
                'thumbnailLink': 'https://static.uahirise.org/images/2010/thumb/PSP_001374_1805.jpg'
            }
        ]

    def test_getImageDetails(self):
        valid_image_id = 'ESP_072116_1740'
        invalid_image_id = 'ESP_072116_1940'

        expected_imageDeets = {
            'title': 'Sinuous Ridge in Aeolis Dorsa',
            'image_id': 'ESP_072116_1740',
            'latitude': '-5.731°',
            'longitude': '152.681°'
        }

        # Test with a valid image_id
        image_deets_valid = imageHelpers.getImageDetails(valid_image_id)

        self.assertEqual(image_deets_valid['title'], expected_imageDeets['title'])
        self.assertEqual(image_deets_valid['image_id'], expected_imageDeets['image_id'])
        self.assertEqual(image_deets_valid['latitude'], expected_imageDeets['latitude'])
        self.assertEqual(image_deets_valid['longitude'], expected_imageDeets['longitude'])

        # Test with an invalid image_id
        image_deets_invalid = imageHelpers.getImageDetails(invalid_image_id)

        self.assertIn('error', image_deets_invalid)
        self.assertEqual(image_deets_invalid['error'], 'Invalid image ID.')

    def test_getMaxPagenumber(self):
        twolinks = [
            '/results.php?keyword=&longitudes=&lon_beg=0&lon_end=8&latitudes=&lat_beg=-20&lat_end=1&solar_all=true&solar_spring=false&solar_summer=false&solar_fall=false&solar_winter=false&solar_equinox=false&solar_equinox_dist=5&solar_solstice=false&solar_solstice_dist=5&solar_beg=&solar_end=&image_all=true&image_anaglyphs=false&image_dtm=false&image_caption=false&order=WP.release_date&science_theme=&page=2',
            '/results.php?keyword=&longitudes=&lon_beg=0&lon_end=8&latitudes=&lat_beg=-20&lat_end=1&solar_all=true&solar_spring=false&solar_summer=false&solar_fall=false&solar_winter=false&solar_equinox=false&solar_equinox_dist=5&solar_solstice=false&solar_solstice_dist=5&solar_beg=&solar_end=&image_all=true&image_anaglyphs=false&image_dtm=false&image_caption=false&order=WP.release_date&science_theme=&page=9'
        ]
        expected_max_page_number = 9

        max_page_number = imageHelpers.getMaxPagenumber(twolinks)

        self.assertEqual(max_page_number, expected_max_page_number)
    
    def test_findLinkById(self):
        link_list = [
            'https://static.uahirise.org/images/2010/thumb/PSP_001374_1805.jpg',
            'https://static.uahirise.org/images/2016/thumb/ESP_045311_1810.jpg',
            'https://static.uahirise.org/images/2012/thumb/ESP_028564_1805.jpg',
            'https://static.uahirise.org/images/2012/thumb/ESP_028630_1805.jpg',
            'https://static.uahirise.org/images/2016/thumb/ESP_047869_1810.jpg'
        ]

        id_exists = 'ESP_047869_1810'
        id_not_exists = 'ESP_999999_9999'

        # Test case for id that exists in the link_list
        result_exists = imageHelpers.findLinkById(link_list, id_exists)
        self.assertEqual(result_exists, 'https://static.uahirise.org/images/2016/thumb/ESP_047869_1810.jpg')

        # Test case for id that does not exist in the link_list
        result_not_exists = imageHelpers.findLinkById(link_list, id_not_exists)
        self.assertIsNone(result_not_exists)
    
    def test_scrapeImages(self):
        link = 'https://www.uahirise.org/results.php?keyword=&longitudes=&lon_beg=7&lon_end=8&latitudes=&lat_beg=0&lat_end=1&solar_all=true&solar_spring=false&solar_summer=false&solar_fall=false&solar_winter=false&solar_equinox=false&solar_equinox_dist=5&solar_solstice=false&solar_solstice_dist=5&solar_beg=&solar_end=&image_all=true&image_anaglyphs=false&image_dtm=false&image_caption=false&order=WP.release_date&science_theme=&page='
        pageNum = 1
        maxPage = None
        images = []

        imageHelpers.scrapeImages(link, pageNum, maxPage, images)

        sorted_expected_images = sorted(self.expected_images, key=lambda x: x['image_id'])
        sorted_images = sorted(images, key=lambda x: x['image_id'])

        self.assertEqual(len(sorted_images), len(sorted_expected_images))
        for image, expected_image in zip(sorted_images, sorted_expected_images):
            # self.assertDictEqual(image, expected_image)
            self.assertEqual(image['title'], expected_image['title'])
            self.assertEqual(image['image_id'], expected_image['image_id'])
            self.assertEqual(image['latitude'], expected_image['latitude'])
            self.assertEqual(image['longitude'], expected_image['longitude'])
            self.assertEqual(image['thumbnailLink'], expected_image['thumbnailLink'])
    
    def test_scrapeImageData(self):

        images = imageHelpers.scrapeImageData(self.json_data)

        for expected_image in self.expected_images:
            image_id = expected_image["image_id"]
            matching_images = [image for image in images if image["image_id"] == image_id]
            self.assertTrue(len(matching_images) > 0, f"No image found for image_id: {image_id}")

            matching_image = matching_images[0]
            self.assertEqual(matching_image["title"], expected_image["title"])
            self.assertEqual(matching_image["latitude"], expected_image["latitude"])
            self.assertEqual(matching_image["longitude"], expected_image["longitude"])
            self.assertEqual(matching_image["thumbnailLink"], expected_image["thumbnailLink"])
    
    def test_getImagesLink(self):

        expected_link = "https://www.uahirise.org/results.php?keyword=&longitudes=&lon_beg=7&lon_end=8&latitudes=&lat_beg=0&lat_end=1&solar_all=true&solar_spring=false&solar_summer=false&solar_fall=false&solar_winter=false&solar_equinox=false&solar_equinox_dist=5&solar_solstice=false&solar_solstice_dist=5&solar_beg=&solar_end=&image_all=true&image_anaglyphs=false&image_dtm=false&image_caption=false&order=WP.release_date&science_theme=&page="

        link = imageHelpers.getImagesLink(self.json_data)
        self.assertEqual(link, expected_link)

    def test_getImgLink(self):
        id = "ESP_072116_1740"
        expected_link = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/ORB_072100_072199/ESP_072116_1740/ESP_072116_1740_RED.NOMAP.browse.jpg?=&="

        link = imageHelpers.getImgLink(id)

        self.assertEqual(link, expected_link)
    
    
#     def test_collectImageIds(self):
#         htmlString = str('''<div class="milo-container-en">
# <div class="page-labels">Search Results </div>
# <!-- begin the thumbnail table --><table align="center" border="0" cellpadding="0" cellspacing="16" width="972"><tr><td align="center" class="catalog-cell-images" valign="top" width="243"><a href="ESP_047869_1810"><img alt="Contrasting Thermal Inertia Unit" border="0" height="117" src="https://static.uahirise.org/images/2016/thumb/ESP_047869_1810.jpg" width="172"/></a><br/><a class (ESP_047869_1810) </a><br/>Lat: 0.9° Long: 7.7°<td align="center" class="catalog-cell-images" valign="top" width="243"><a href=" border="0" height="117" src="https://static.uahirise.org/images/2016/thumb/ESP_045311_1810.jpg" width="172"/></a><br/><a class (ESP_045311_1810) </a><br/>Lat: 0.9° Long: 7.2°<td align="center" class="catalog-cell-images" valign="top" width="243"><a href=" border="0" height="117" src="https://static.uahirise.org/images/2012/thumb/ESP_028630_1805.jpg" width="172"/></a><br/><a class (ESP_028630_1805) </a><br/>Lat: 0.7° Long: 7.4°<td align="center" class="catalog-cell-images" valign="top" width="243"><a href=" border="0" height="117" src="https://static.uahirise.org/images/2012/thumb/ESP_028564_1805.jpg" width="172"/></a><br/><a class (ESP_028564_1805) </a><br/>Lat: 0.7° Long: 7.4°</td></td></td></td></tr><tr><td align="center" class="catalog-cell-images" valign="top" width="243"><a href="PSP_001374_1805"><img alt="Layering in an Exhumed Crater at Meridiani Planum" border="0" height="117" src="https://static.uahirise.org/images/2010/thumb/PSP_001374_1805.jpg" width="172"/></a><br/><a class="cells" href="PSP_001374_1805">Layering in an Exhumed Crater at Meridiani Planum (PSP_001374_1805) </a><br/>Lat: 0.7° Long: 7.4°</td></tr><tr><td align="left" valign="top" width="243"><br/>  <br/><br/></td><td align="center" class="page-numbers" colspan="2" valign="top"><br/>Page 1 of 1 pages (5 images)<br/><br/></td><td align="right" valign="top" width="243"><br/>  <br/><br/></td></tr> </table>       
# <!-- end the thumbnail table -->
# </div>''')

#         expected_uniqueImageIds = [
#             'ESP_028564_1805', 'ESP_028630_1805', 'ESP_045311_1810', 'ESP_047869_1810', 'PSP_001374_1805'
#         ]

#         expected_uniqueImageThumbnails = [
#             'https://static.uahirise.org/images/2010/thumb/PSP_001374_1805.jpg',
#             'https://static.uahirise.org/images/2012/thumb/ESP_028564_1805.jpg',
#             'https://static.uahirise.org/images/2012/thumb/ESP_028630_1805.jpg',
#             'https://static.uahirise.org/images/2016/thumb/ESP_045311_1810.jpg',
#             'https://static.uahirise.org/images/2016/thumb/ESP_047869_1810.jpg'
#         ]

#         uniqueImageIds, uniqueImageThumbnails = imageHelpers.collectImageIds(htmlString)

#         self.assertEqual(uniqueImageIds, expected_uniqueImageIds)
#         self.assertEqual(uniqueImageThumbnails, expected_uniqueImageThumbnails)

# Test 1
class TestDeleteFiles(TestCase):

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
        context_enhanced_api.deleteFiles(self.file_list)

        # Assert that the files are deleted
        for file in self.file_list:
            self.assertFalse(os.path.exists(file))


class TestDeleteFolder(TestCase):
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
        context_enhanced_api.deleteFolder(self.temp_dir)

        # Assert that the folder no longer exists
        self.assertFalse(os.path.exists(self.temp_dir))

class TestGetRotatedImage(TestCase):

    def setUp(self):
        self.root = "api/tests/images/"
        self.test_image = "test_rotation.jpg"
        self.rotated_image = "test_rotated_image.jpg"

    def test_getRotatedImage(self):

        # Call the function under test
        result = context_enhanced_api.getRotatedImage(self.test_image, self.root)

        rotated_image = cv2.imread(self.root + self.rotated_image, cv2.COLOR_BGR2GRAY)
        _, img_encoded = cv2.imencode('.jpeg', rotated_image)

        # Assert the result
        self.assertEqual(result.all(), img_encoded.all())

    def tearDown(self):
        # Clean up the test image and rotated image
        os.remove(self.root + self.rotated_image)

class TestGetSegmentationResult(TestCase):

    def setUp(self):
        self.root = "api/tests/images/"
        self.test_image = "test_segmentation.jpg"
        self.segmented_image = "segmented_image.png"

    def test_getSegmentationResult(self):

        # Call the function under test
        result = context_enhanced_api.getSegmentationResult(self.test_image, self.root)

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