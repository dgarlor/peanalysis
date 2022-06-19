import sys
import requests
import json
import time
import os

from peUtils import writeJson, getToken

datafolder = "data\imtstatistiques"
os.makedirs(datafolder,exist_ok=True)




##
##   obtain a token before calling the API for the first time
##
token = getToken()
api_call_headers = {'Authorization': 'Bearer ' + token}
stats_offres_url = "https://api.emploi-store.fr/partenaire/infotravail/v1/datastore_search_sql"



def sqlRequest(sqlrequest):

      ### IMT - Statistiques sur les offres et demandes => SIZE: 269220
      # GET https://api.emploi-store.fr/partenaire/infotravail/v1/datastore_search_sql?sql=SELECT * FROM "266f691f-bce8-4443-808e-8e5aa125cf17" WHERE "ROME_PROFESSION_CARD_CODE" LIKE 'A1101' AND "AREA_TYPE_CODE" = 'B' AND "AREA_CODE" = '5311'
      api_call_response = requests.get(stats_offres_url,
            headers=api_call_headers, verify=True,
            params={"sql":sqlrequest})
      time.sleep(1)
      if api_call_response.status_code != 200:
            print(" -- ERROR in request",api_call_response.status_code)
            print(api_call_response.reason)
            return None
      print(json.dumps(api_call_response.json(), indent=4))
      jsonresponse = api_call_response.json()
      success = jsonresponse["success"]
      if success:
            return jsonresponse["result"]
      else:
            return None

# sqlrequest = f'SELECT COUNT(*) FROM "266f691f-bce8-4443-808e-8e5aa125cf17"'
# count_response = sqlRequest(sqlrequest)
# size = int(count_response["records"][0]["count"])
# print(size)
size =269220
steps = (size + 99) // 100
for s in range(steps):
      sqlrequest = f'SELECT * FROM "266f691f-bce8-4443-808e-8e5aa125cf17" LIMIT 100 OFFSET {s*100}'
      sqlrequest = f'SELECT COUNT(*) FROM "266f691f-bce8-4443-808e-8e5aa125cf17" WHERE "ROME_PROFESSION_CARD_CODE" LIKE \'D1505\' LIMIT 100 OFFSET {s*100}'
      part = sqlRequest(sqlrequest)
      break
