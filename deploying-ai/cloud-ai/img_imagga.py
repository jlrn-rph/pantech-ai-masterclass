# OBJECT DETECTION USING IMAGGA'S IMAGE TAGGING API

# import libraries
import requests
import json
from keyImagga import AUTHORIZATION

# setup authentication and API keys
url = "https://api.imagga.com/v2/tags"

querystring = {"image_url":"https://businessmirror.com.ph/wp-content/uploads/2019/05/top01a-050419.jpg"}

headers = {
    'accept': "application/json",
    'authorization': AUTHORIZATION
}

# get data from url
response = requests.request("GET", url, headers=headers, params=querystring)
data = json.loads(response.text.encode("ascii"))

# print tags
for i in range(6):
    tag = data["result"]["tags"][i]["tag"]["en"]
    print(tag)
