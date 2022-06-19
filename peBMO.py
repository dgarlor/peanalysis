import sys
import requests
import json
import time
import os

from peUtils import writeJson, getToken

datafolder = "data/bmo"
os.makedirs(datafolder,exist_ok=True)


##
##   obtain a token before calling the API for the first time
##
token = getToken()
api_call_headers = {'Authorization': 'Bearer ' + token}


# 47dbbaba-c983-47df-bd9c-eaeec14bd834 // Statistiques sur le marché du travail
# ##Ce jeu de données référence les informations statistiques sur le marché du travail : données statistiques sur la demande d’emploi.
package_id = "521b5b3e-9bcc-4e8e-b6f1-2a0a0899989a"


package_file = f"data/packages/package_{package_id}_BMO.json"
package = json.load(open(package_file))
np = len(package["result"]["resources"])
for r in package["result"]["resources"][np//2:]:
  ressource_id = r["id"]
  pe_source = r["pe_source"]
  ## show resource
  ##GET https://api.emploi-store.fr/partenaire/infotravail/v1/resource_show?id=[Identifiant d'une ressource]
  ressource_file = f"{datafolder}/res_{pe_source}.json"
  if not os.path.exists(ressource_file):
    resource_show_url = "https://api.emploi-store.fr/partenaire/infotravail/v1/resource_show"
    api_call_response = requests.get(resource_show_url,
      headers=api_call_headers, verify=True,
      params={"id":ressource_id})
    time.sleep(1)
    ressource_info = api_call_response.json()
    writeJson(ressource_file, ressource_info)
  else:
    ressource_info = json.load(open(ressource_file))
  
  #print(json.dumps(ressource_info, indent=4))
  print(r["pe_source"]," -- ",r["name"],r["id"]," SIZE:",r["size"])
  ressource_size = r["size"] if r["size"] else 1000000

  remaining = int(ressource_size)
  offset = 0
  step = 100
  while remaining > 0:
    print(" == Remaining ",remaining,"/",ressource_size)
    res_part_file = f"{datafolder}/rp_{ressource_id}_o{offset}.json"
    if not os.path.exists(res_part_file):
      # lire ressource
      ## GET https://api.emploi-store.fr/partenaire/infotravail/v1/datastore_search?resource_id=[Identifiant d'une ressource]
      resource_show_url = "https://api.emploi-store.fr/partenaire/infotravail/v1/datastore_search"
      api_call_response = requests.get(resource_show_url,
                headers=api_call_headers, verify=True,
                params={"resource_id":ressource_id, "limit":step,"offset":offset}) #"limit":100,"offset":100

      res_part = api_call_response.json()
      writeJson(res_part_file, res_part)
    else:
      res_part = json.load(open(res_part_file))
    if "records" not  in res_part["result"]:
      print(" ERROR: in offest",offset)
      sys.exit(1)
    print(" Read: ",len(res_part["result"]["records"]))
    if len(res_part["result"]["records"]) == 0:
      break
    #print(json.dump(res_part))
    remaining -= step
    offset += step

