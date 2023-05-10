from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import requests
from django.http import HttpResponse
import os
from . import context_enhanced_api
from bs4 import BeautifulSoup
import re

@api_view(['GET'])
def getImageDetails(request, image_id):
    link = f'https://www.uahirise.org/{image_id}'
    response = requests.get(link)

    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    observation_title = soup.find('span', class_='observation-title-milo').text

    latitude_pattern = r'Latitude \(centered\)<\/strong><br \/>(-?\d+\.\d+)&deg;'
    longitude_pattern = r'Longitude \(East\)<\/strong><br \/>(-?\d+\.\d+)&deg;'
    latitude_match = re.search(latitude_pattern, html_content)
    longitude_match = re.search(longitude_pattern, html_content)

    if latitude_match and longitude_match:
        latitude = latitude_match.group(1) + '°'
        longitude = longitude_match.group(1) + '°'

    image = {'title': observation_title, 'image_id': image_id, 'latitude': latitude, 'longitude': longitude}
    return Response(image)


@api_view(['GET'])
def getImage(request, image_id):
    # response = requests.get('https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/ORB_016900_016999/ESP_016934_1770/ESP_016934_1770_RED.NOMAP.browse.jpg?=&=')
    # image_data = response.content
    # return HttpResponse(image_data, content_type='image/jpeg')
    # return response
    link = getImgLink(image_id)
    response = requests.get(link)
    image_data = response.content
    file_name = saveImage(image_id, image_data)
    img_encoded = context_enhanced_api.getRotatedImage(file_name)
    response = HttpResponse(img_encoded.tobytes(), content_type='image/jpeg')
    return response
    # return HttpResponse(image_data, content_type='image/jpeg')

@api_view(['GET'])
def getImageSegmentation(request, image_id):
    link = getImgLink(image_id)
    # image_data = {'id': image_id, 'link': link}
    # return Response(image_data)
    response = requests.get(link)
    image_data = response.content
    file_name = saveImage(image_id, image_data)
    
    img_encoded = context_enhanced_api.getSegmentationResult(file_name)
    response = HttpResponse(img_encoded.tobytes(), content_type='image/png')
    return response

def saveImage(image_id, image_data):
    file_name = f"{image_id}.jpg"
    file_path = os.path.join("api\images", file_name)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return file_name


def getImgLink(id):
    idSegments = id.split('_')
    link = ''
    if len(idSegments) > 1:
        imgType = idSegments[0]
        lowerBound = idSegments[1][:4] + '00'
        upperBound = idSegments[1][:4] + '99'
        link = f'https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/{imgType}/ORB_{lowerBound}_{upperBound}/{id}/{id}_RED.NOMAP.browse.jpg?=&='
    return link