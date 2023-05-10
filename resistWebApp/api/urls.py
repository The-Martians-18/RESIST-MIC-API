from django.urls import path
from . import views

urlpatterns = [
    # path('', views.get_image),
    path('image/<str:image_id>/', views.getImage, name='get_image'),
    path('imageseg/<str:image_id>/', views.getImageSegmentation, name='get_image_segmentation'),
]
