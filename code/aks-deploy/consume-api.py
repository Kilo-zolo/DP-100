# ----------------------------------------------------------------------
# Consume the webservice using API end points using "requests" library
#
# Steps to use requests module in python
# 1. Get scoring URI
# 2. Prepare the input data and create JSON
# 3. Prepare the headers with authorization key
# 4. Make a POST request to API using URI, headers and data
# 5. Collect the response and deserialize the JSON
# ----------------------------------------------------------------------

import requests
import json

# ------------------------------------------------------
# Set the URI for the web service
# ------------------------------------------------------
scoring_uri = 'http://52.150.36.193:80/api/v1/service/adultincome-service/score'


# ------------------------------------------------------
# Prepare the input data and create the serialized JSON
# ------------------------------------------------------
x_new = {'age':[21],
         'wc':['Private'],
         'education':['HS-grad'],
         'marital status':['Never-married'],
         'race':['White'],
         'gender':['Male'],
         'hours per week':[30]}

# Convert the input data to a serialized JSON
json_data = json.dumps({"data": x_new})



# ---------------------------------------------------
# Create the headers with authorization key
# ---------------------------------------------------
key = 'WxpCLzMGSPz3LN6uhYF3TMIeBdogT0Vp'

# Set the content type and authorization
headers = {'Content-Type': 'application/json',
           'Authorization' : f'Bearer {key}'}


# ---------------------------------------------------
# Make the request using POST method and collect the response
# ---------------------------------------------------
response = requests.post(scoring_uri, json_data, headers=headers)
predicted_classes = json.loads(response.json())

print('\n', predicted_classes)













