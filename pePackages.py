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


packages_file = f"{datafolder}/packages.json"
if not os.path.exists(packages_file):
      digidata_url = "https://api.emploi-store.fr/partenaire/infotravail/v1/organization_show"
      api_call_response = requests.get(digidata_url,
            headers=api_call_headers, verify=False,
            params={"id":"digidata"})
      time.sleep(1)
      writeJson(packages_file, api_call_response.json())

packages = json.load(open(packages_file))
liste = packages["result"]["packages"]


for l in liste:
      print(l["id"],"//",l["title"],"====",l["notes"])

      pid = l["id"]
      package_file = f"{datafolder}/package_{pid}.json"
      if not os.path.exists(package_file):

            # SHOW PACKAGE
            ##GET https://api.emploi-store.fr/partenaire/infotravail/v1/package_show?id=[Identifiant du jeu de données]
            package_url = "https://api.emploi-store.fr/partenaire/infotravail/v1/package_show"
            api_call_headers = {'Authorization': 'Bearer ' + token}
            api_call_response = requests.get(package_url,
                  headers=api_call_headers, verify=False,
                  params={"id":l["id"]})
            time.sleep(1)
            writeJson(package_file, api_call_response.json())

      package = json.load(open(package_file))
      for r in package["result"]["resources"]:
            print(r["pe_source"]," -- ",r["name"],r["id"])
            idr = r["id"]
