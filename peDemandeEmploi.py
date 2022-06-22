import sys
import requests
import json
import time
import os

from peUtils import writeJson, getToken

datafolder = "data\demandeEmploi"
os.makedirs(datafolder,exist_ok=True)


##
##   obtain a token before calling the API for the first time
##
token = getToken()
api_call_headers = {'Authorization': 'Bearer ' + token}


# 47dbbaba-c983-47df-bd9c-eaeec14bd834 // Statistiques sur le marché du travail
# ##Ce jeu de données référence les informations statistiques sur le marché du travail : données statistiques sur la demande d’emploi.
package_id = "47dbbaba-c983-47df-bd9c-eaeec14bd834"


package_file = f"data/packages/package_{package_id}.json"
package = json.load(open(package_file))
np = len(package["result"]["resources"])
for r in package["result"]["resources"]:
  ressource_id = r["id"]
  pe_source = r["pe_source"]
  # if pe_source.count("2012") == 0:
  #   continue
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

  if pe_source.count("DEMANDE") == 0:
    print(" -- SKIPPING")
    continue
  
  #print(json.dumps(ressource_info, indent=4))
  print(r["pe_source"]," -- ",r["name"],r["id"]," SIZE:",r["size"])
  sourceDir = f"{datafolder}/{pe_source}"
  os.makedirs(sourceDir,exist_ok=True)

  ressource_size = r["size"]

  remaining = int(ressource_size)
  offset = 0
  step = 100
  retries = 0
  while remaining > 0:
    try:
      res_part_file = f"{sourceDir}/rp_{ressource_id}_o{offset}.json"
      if os.path.exists(res_part_file):
        remaining -= step
        offset += step
        continue

      if offset % 10000 == 0:
        print(" == Remaining ",remaining,"/",ressource_size)

      if not os.path.exists(res_part_file):
        # lire ressource
        ## GET https://api.emploi-store.fr/partenaire/infotravail/v1/datastore_search?resource_id=[Identifiant d'une ressource]
        resource_show_url = "https://api.emploi-store.fr/partenaire/infotravail/v1/datastore_search"
        api_call_response = requests.get(resource_show_url,
                  headers=api_call_headers, verify=True,
                  params={"resource_id":ressource_id, "limit":step,"offset":offset}) #"limit":100,"offset":100
        time.sleep(0.1)
        if api_call_response.status_code != 200:
          print(" ERROR: ",api_call_response.status_code, api_call_response.reason)
          if retries < 2:
            token = getToken()
            api_call_headers = {'Authorization': 'Bearer ' + token}
            retries += 1
            continue
          else:
            print(" -TOO MUCH REPLIES")
            sys.exit(1)
        retries = 0
        res_part = api_call_response.json()
        if len(res_part["result"]["records"]) == 0:
          print(" EMPTY JSON")
          break      
        writeJson(res_part_file, res_part)
      else:
        res_part = json.load(open(res_part_file))
      if "records" not  in res_part["result"]:
        print(" ERROR: in offest",offset)
        sys.exit(1)

      if len(res_part["result"]["records"]) == 0:
        break
      #print(json.dump(res_part))
      remaining -= step
      offset += step
    except Exception as e:
      print(" -- ERROR RETRYING!")
      print(e)

