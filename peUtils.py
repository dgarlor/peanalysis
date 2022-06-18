import sys
import requests
import json

from private import client_id, client_secret

def writeJson(filename, structure):
  with open(filename,"w") as pout:
    pout.write(json.dumps(structure, indent=4))

def getToken():
  """ Get new token"""
  auth_server_url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token"
  token_req_payload = {'grant_type': 'client_credentials','client_id':client_id,'client_secret':client_secret,'scope':'api_infotravailv1'}

  token_response = requests.post(auth_server_url,
        data=token_req_payload, verify=True, allow_redirects=True,
        params={'realm':'/partenaire'},
        headers={"Content-Type":"application/x-www-form-urlencoded"})

  if token_response.status_code !=200:
    print("Failed to obtain token from the OAuth 2.0 server", file=sys.stderr)
    print(token_response.content,token_response.status_code)
    sys.exit(1)

  print("Successfuly obtained a new token",token_response.text,token_response)
  return token_response.json()['access_token']

