from lxml import html
import requests
import re
import json

domain = 'https://openi.nlm.nih.gov/'
url = 'https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x'
regex = re.compile(r"var oi = (.*);")
final_data = {}
img_no = 0


def extract(url):
    global img_no
    img_no += 1
    r = requests.get(url)
    tree = html.fromstring(r.text)

    div = tree.xpath('//table[@class="masterresultstable"]\
        //div[@class="meshtext-wrapper-left"]')[0]

    typ = div.xpath('.//strong/text()')[0]
    items = div.xpath('.//li/text()')
    img = tree.xpath('//img[@id="theImage"]/@src')[0]

    final_data[img_no] = {}
    final_data[img_no]['type'] = typ
    final_data[img_no]['items'] = items
    final_data[img_no]['img'] = domain + img
    print final_data[img_no]


def main():
    r = requests.get(url)
    tree = html.fromstring(r.text)

    script = tree.xpath('//script[@language="javascript"]/text()')[0]

    json_string = regex.findall(script)[0]
    json_data = json.loads(json_string)

    print 'extract'
    links = [domain + x['nodeRef'] for x in json_data]
    for link in links:
        extract(link)

if __name__ == '__main__':
    main()
    # print final_data
    with open('data.json', 'w') as f:
        json.dump(final_data, f)
