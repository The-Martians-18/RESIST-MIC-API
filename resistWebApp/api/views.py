from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import requests
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseNotFound
import os
from . import context_enhanced_api, imageHelpers

@api_view(['GET'])
def getImageDetails(request, image_id):
    imageDeets = imageHelpers.getImageDetails(image_id)
    if 'error' in imageDeets:
        return Response(imageDeets, status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response(imageDeets)

@api_view(['GET'])
def getImage(request, image_id):
    link = imageHelpers.getImgLink(image_id)
    
    if link is None:
        return HttpResponseBadRequest(content_type='application/json',
                                      content='{"error": "Invalid image ID."}')
    
    response = requests.get(link)
    
    if response.status_code == 404:
        return HttpResponseNotFound(content_type='application/json',
                                    content='{"error": "Image not found. Invalid image ID."}')
    
    image_data = response.content
    file_name = imageHelpers.saveImage(image_id, image_data)
    img_encoded = context_enhanced_api.getRotatedImage(file_name)
    response = HttpResponse(img_encoded.tobytes(), content_type='image/jpeg')
    return response

@api_view(['GET'])
def getImageSegmentation(request, image_id):
    link = imageHelpers.getImgLink(image_id)
    
    if link is None:
        return HttpResponseBadRequest(content_type='application/json',
                                      content='{"error": "Invalid image ID."}')
    
    response = requests.get(link)
    
    if response.status_code == 404:
        return HttpResponseNotFound(content_type='application/json',
                                    content='{"error": "Image not found. Invalid image ID."}')
    image_data = response.content
    file_name = imageHelpers.saveImage(image_id, image_data)
    
    img_encoded = context_enhanced_api.getSegmentationResult(file_name)
    response = HttpResponse(img_encoded.tobytes(), content_type='image/png')
    return response

@api_view(['POST'])
def getImagesInBoundary(request):
    json_data = request.data
    # Access the JSON data
    images = imageHelpers.scrapeImageData(json_data)
    boundaries = {'images': images}

    return Response(boundaries)