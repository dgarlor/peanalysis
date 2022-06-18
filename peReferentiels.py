import sys
import requests
import json
import time
import os

from peUtils import writeJson, getToken
datafolder = "data"

##
##   obtain a token before calling the API for the first time
##
token = getToken()
api_call_headers = {'Authorization': 'Bearer ' + token}

# ##
# ##   REFERENTIELS!!!
# ##
referentiels_file = f"{datafolder}/referentiels.json"
if not os.path.exists(referentiels_file):
  referentiels_url ='https://api.emploi-store.fr/partenaire/infotravail/v1/resource_search'
  query = {"query":"pe_type:reference"}
  
  api_call_response = requests.get(referentiels_url,
        headers=api_call_headers, verify=False,
        params=query)
  time.sleep(1)
  #print(json.dumps(api_call_response.json(), indent=4))
  writeJson(referentiels_file,api_call_response.json())


# Print referentiels
referentiels = json.load(open(referentiels_file))
liste = referentiels["result"]["results"]
for l in liste:
  print(l["pe_source"],"//",l["name"],"====",l["description"],"::",l["size"])
  pe_source = l["pe_source"]

  reffile =  f"{datafolder}/ref_{pe_source}.json"
  # Get all referentiels
  if not os.path.exists(reffile):
    datastore_url = "https://api.emploi-store.fr/partenaire/infotravail/v1/datastore_search"
    api_call_response = requests.get(datastore_url,
          headers=api_call_headers, verify=False,
          params={"resource_id":l["id"]})

    writeJson(reffile,api_call_response.json())
  
    time.sleep(1)
