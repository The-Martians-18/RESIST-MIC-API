from django.urls import path
from . import views

urlpatterns = [
    path('imagedetails/<str:image_id>/', views.getImageDetails, name='get_image_details'),
    path('image/<str:image_id>/', views.getImage, name='get_image'),
    path('imageseg/<str:image_id>/', views.getImageSegmentation, name='get_image_segmentation'),

]
