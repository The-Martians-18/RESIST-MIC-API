from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from unittest.mock import patch

class ImageViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch('api.views.imageDetails.getImageDetails')
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

    @patch('api.views.imagesInBoundary.scrapeImageData')
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