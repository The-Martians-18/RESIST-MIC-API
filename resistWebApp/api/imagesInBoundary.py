import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from . import imageDetails

def find_href_values(obj, results):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "@href":
                results.add(value)
            elif isinstance(value, (dict, list)):
                find_href_values(value, results)
    elif isinstance(obj, list):
        for item in obj:
            find_href_values(item, results)

def getMaxPagenumber(twolinks):
    max_page_number = -1
    for link in twolinks:
        parsed_link = urlparse(link)
        query_params = parse_qs(parsed_link.query)
        page_number = query_params.get('page', [])[0] if 'page' in query_params else None
        if page_number is not None and int(page_number) > max_page_number:
            max_page_number = int(page_number)
    return(max_page_number)

def collectImageIds(htmlString):
    soup = BeautifulSoup(htmlString, 'html.parser')
    href_tags = soup.find_all('a')
    href_tags = list(set(href_tags))
    imageIds = []
    imageThumbnails = []
    for tag in href_tags:
        href_value = tag.get('href')
        imageIds.append(href_value)
        img_tag = tag.find('img')
        if img_tag:
            src_link = img_tag.get('src')
            imageThumbnails.append(src_link)
    uniqueImageIds = list(set(imageIds))
    uniqueImageThumbnails = list(set(imageThumbnails))
    return uniqueImageIds, uniqueImageThumbnails

def findLinkById(link_list, id):
    for link in link_list:
        if id in link:
            return link
    return None

def scrapeImages(link, pageNum, maxPage, images):
    nextLink = link + str(pageNum)
    response = requests.get(nextLink)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    cells = list(soup.find_all('div', class_ = 'milo-container-en'))
    imageIds_raw, thmubnailLinks_raw = collectImageIds(str(cells[0]))

    twoLinks = []
    for imageId in imageIds_raw:
        if imageId[:8] == "/results":
            twoLinks.append(imageId)
        else:
            imageDeets = imageDetails.getImageDetails(imageId)
            thumbnailLink = findLinkById(thmubnailLinks_raw, imageId)
            imageDeets["thumbnailLink"] = thumbnailLink
            images.append(imageDeets)
    
    if maxPage is None:
        maxPage = getMaxPagenumber(twoLinks)
    nextPage = pageNum + 1

    if nextPage > maxPage:
        return
    scrapeImages(link, nextPage, maxPage, images)

def scrapeImageData(json_data):
    link = getImagesLink(json_data)
    images = []
    scrapeImages(link, 1, None, images)
    return images

def getImagesLink(json_data):
    latitude_beginning = json_data.get('latitude_beginning')
    latitude_ending = json_data.get('latitude_ending')
    longitude_beginning = json_data.get('longitude_beginning')
    longitude_ending = json_data.get('longitude_ending')
    link = f'https://www.uahirise.org/results.php?keyword=&longitudes=&lon_beg={longitude_beginning}&lon_end={longitude_ending}&latitudes=&lat_beg={latitude_beginning}&lat_end={latitude_ending}&solar_all=true&solar_spring=false&solar_summer=false&solar_fall=false&solar_winter=false&solar_equinox=false&solar_equinox_dist=5&solar_solstice=false&solar_solstice_dist=5&solar_beg=&solar_end=&image_all=true&image_anaglyphs=false&image_dtm=false&image_caption=false&order=WP.release_date&science_theme=&page='
    return link