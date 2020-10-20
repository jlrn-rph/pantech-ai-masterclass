# DATASET GENERATOR - COLLECTING DATASET FROM GOOGLE IMAGES

# import libraries
from simple_image_download import simple_image_download as sid

# initialize variable
response = sid.simple_image_download

# download by providing keyword
response().download('mabel gravity falls', 150)