from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import requests
from django.http import HttpResponse
import os
from . import context_enhanced_api

# @api_view(['GET'])
# def getData(request):
#     person = {'name': 'Gowantha', 'age':28}
#     return Response(person)


@api_view(['GET'])
def getImage(request, image_id):
    # response = requests.get('https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/ORB_016900_016999/ESP_016934_1770/ESP_016934_1770_RED.NOMAP.browse.jpg?=&=')
    # image_data = response.content
    # return HttpResponse(image_data, content_type='image/jpeg')
    # return response
    link = getImgLink(image_id)
    response = requests.get(link)
    image_data = response.content
    return HttpResponse(image_data, content_type='image/jpeg')

@api_view(['GET'])
def getImageSegmentation(request, image_id):
    link = getImgLink(image_id)
    # image_data = {'id': image_id, 'link': link}
    # return Response(image_data)
    response = requests.get(link)
    image_data = response.content

    file_name = f"{image_id}.jpg"
    file_path = os.path.join("api\images", file_name)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    
    img_encoded = context_enhanced_api.getSegmentationResult(file_name)
    response = HttpResponse(img_encoded.tobytes(), content_type='image/png')
    return response

    return HttpResponse(image_data, content_type='image/jpeg')
    image_data = {'id': image_id, 'link': link}
    return Response(image_data)


def getImgLink(id):
    idSegments = id.split('_')
    link = ''
    if len(idSegments) > 1:
        imgType = idSegments[0]
        lowerBound = idSegments[1][:4] + '00'
        upperBound = idSegments[1][:4] + '99'
        link = f'https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/{imgType}/ORB_{lowerBound}_{upperBound}/{id}/{id}_RED.NOMAP.browse.jpg?=&='
    return link