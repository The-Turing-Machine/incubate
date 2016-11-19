from lxml import html
import requests
import re

url = ''

r = requests.get(url)
tree = html.fromstring(r.text)
imgs = tree.xpath('//img[contains(@class, "brand-img")]/@src')
