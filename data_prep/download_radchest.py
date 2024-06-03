# Thank you to https://github.com/zenodo/zenodo/issues/1888#issuecomment-793348649

import requests

ACCESS_TOKEN = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTcxMjc2MzEwNCwiZXhwIjoxNzE3ODkxMTk5fQ.eyJpZCI6ImViY2ZlMGYxLTVkNDAtNDY2MC05MDkzLWYyYzVmM2M2YWIyNCIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZDBhNTY4ZjM4YTJjMmIzNGQyM2RjNzQ4ZWM1YjE3NCJ9.hDCEtgzLkno7_2vaCawDDMRJ7f4Gt_XtnMKw8KfcGhokw2mj5kuk-flOVcdoQ4uPXKi9Ih6M2ag1CEBULTIfig"
record_id = "6406114"

r = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'token': ACCESS_TOKEN})
download_urls = [f['links']['self'] for f in r.json()['files']]
filenames = [f['key'] for f in r.json()['files']]

print(r.status_code)
print(download_urls)


output_dir = "/dtu/p1/johlau/Thesis-Synthex/data/RAD-ChestCT/"

for filename, url in zip(filenames, download_urls):
    print("Downloading:", filename)
    r = requests.get(url, params={'token': ACCESS_TOKEN})
    with open(output_dir+filename, '+wb') as f:
        f.write(r.content)