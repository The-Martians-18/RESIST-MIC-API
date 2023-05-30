import requests
from bs4 import BeautifulSoup
import re

def getImageDetails(image_id):
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

    imageDeets = {'title': observation_title, 'image_id': image_id, 'latitude': latitude, 'longitude': longitude}
    return imageDeets
