from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import requests
from django.http import HttpResponse
import os
from . import context_enhanced_api
from . import imagesInBoundary, imageDetails

@api_view(['GET'])
def getImageDetails(request, image_id):
    imageDeets = imageDetails.getImageDetails(image_id)
    return Response(imageDeets)

@api_view(['GET'])
def getImage(request, image_id):
    link = getImgLink(image_id)
    response = requests.get(link)
    image_data = response.content
    file_name = saveImage(image_id, image_data)
    img_encoded = context_enhanced_api.getRotatedImage(file_name)
    response = HttpResponse(img_encoded.tobytes(), content_type='image/jpeg')
    return response

@api_view(['GET'])
def getImageSegmentation(request, image_id):
    link = getImgLink(image_id)
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

@api_view(['POST'])
def getImagesInBoundary(request):
    json_data = request.data
    # Access the JSON data
    images = imagesInBoundary.scrapeImageData(json_data)
    boundaries = {'images': images}

    return Response(boundaries)