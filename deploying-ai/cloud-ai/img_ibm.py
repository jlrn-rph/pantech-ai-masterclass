# OBJECT DETECTION USING IBM WATSON'S VISUAL RECOGNITION API

# import libraries
import json
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from keyIbm import AUTHENTICATOR, URL

# setup authentication and API keys
authenticator = IAMAuthenticator(AUTHENTICATOR)
visual_recognition = VisualRecognitionV3(
    version='2018-03-19',
    authenticator=authenticator
)   

visual_recognition.set_service_url(URL)

# image url
url = 'https://businessmirror.com.ph/wp-content/uploads/2019/05/top01a-050419.jpg'

# identify url's classes
classes_result = visual_recognition.classify(url=url).get_result()
print(json.dumps(classes_result, indent=2))
