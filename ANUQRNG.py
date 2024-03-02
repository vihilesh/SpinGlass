# Define new data to create

import base64
import json
from math import floor
import os
import requests


form_data = {
    "repeats": "yesrepeat",
    "set_num": 50,
    "rep_num": 500,
    "min_num": 0,
    "max_num": 5,
    "action": "dice_action",
    "dice_nonce_field": "d23b392c6d",
    "_wp_http_referer": "/dice-throw/"
}

headers = {
'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
'Accept': 'application/json, text/javascript, */*; q=0.01',
'Accept-Encoding': 'gzip, deflate, br, zstd',
'Accept-Language': 'en-US,en;q=0.9',
'Connection': 'keep-alive',
'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
'Host':'qrng.anu.edu.au',
'Origin': 'https://qrng.anu.edu.au',
'Referer': 'https://qrng.anu.edu.au/dice-throw/',
'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
'Sec-Ch-Ua-Mobile': '?0',
'Sec-Ch-Ua-Platform': "Windows",
'Sec-Fetch-Dest': "empty",
'Sec-Fetch-Mode':"cors",
'Sec-Fetch-Site':'same-origin',
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
'X-Requested-With':'XMLHttpRequest'
}





# The API endpoint to communicate with
url_post = "https://qrng.anu.edu.au/wp-admin/admin-ajax.php"

# A POST request to tthe API
post_response = requests.post(url_post, data=form_data, headers=headers, verify=False)

jsontxt = json.loads(post_response.json())
output = (jsontxt["output"])
#print(type(output))
with open("response.txt", "a") as f:
    for _list in output:
        for _string in _list:
            #f.seek(0)
            f.write(str(_string) + '\n')

    

    #with open("response.txt", "w") as f:
    #    f.write(jsontxt.get('output'))
