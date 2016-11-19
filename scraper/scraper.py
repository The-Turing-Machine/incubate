from lxml import html
import requests
import re
import json

url = 'https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x'
regex = re.compile(r"var oi = (.*);")

r = requests.get(url)
tree = html.fromstring(r.text)

script = tree.xpath('//script[@language="javascript"]/text()')[0]

json_string = regex.findall(script)[0]
json_data = json.loads(json_string)

with open('data.json', 'w') as f:
    json.dump(json_data, f)
