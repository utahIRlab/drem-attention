import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import ssl
import csv
import sys
import time
import random

def GET_UA():
    uastrings = ["Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",\
                "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.72 Safari/537.36",\
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.25 (KHTML, like Gecko) Version/8.0 Safari/600.1.25",\
                "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",\
                "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",\
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",\
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/600.1.17 (KHTML, like Gecko) Version/7.1 Safari/537.85.10",\
                "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",\
                "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",\
                "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.104 Safari/537.36"\
                ]
 
    return random.choice(uastrings)

def scrape(url, counter):
	# For ignoring SSL certificate errors
	# ctx = ssl.create_default_context()
	# ctx.check_hostname = False
	# ctx.verify_mode = ssl.CERT_NONE

	product_json = dict()

	user_agent = GET_UA()
	values = {'name': 'Mary Foord',
          'location': 'Northampton',
          'language': 'Python'}
	headers = {'User-Agent': user_agent}

	data = urllib.parse.urlencode(values)
	data = data.encode('ascii')
	req = urllib.request.Request(url, data, headers)

	try:
		html = urllib.request.urlopen(req).read()
		time.sleep(1)
		# html = urllib.request.urlopen(url).read()
		soup = BeautifulSoup(html, 'html5lib')
		# This block of code will help extract the Prodcut Title of the item

		for spans in soup.findAll('span', attrs={'id': 'productTitle'}):
			name_of_product = spans.text.strip()
			product_json['title'] = name_of_product
			break

		# This block of code will help extract the image of the item

		for divs in soup.findAll('div', attrs={'class': 'imgTagWrapper', 'id': 'imgTagWrapperId'}):
			for img_tag in divs.findAll('img', attrs={'id': 'landingImage'
													  }):
				# product_json['image'] = img_tag['src']
				product_json['image'] = img_tag['data-a-dynamic-image'][img_tag['data-a-dynamic-image'].find('"')+1:img_tag['data-a-dynamic-image'].find('":')]
				break

		# This block of code will help extract top specifications and details of the product

		product_json['description'] = []
		for div in soup.findAll('div', attrs={'id': 'feature-bullets'}):
			for li_tags in div.findAll('li'):
				for spans in li_tags.findAll('span',
											 attrs={'class': 'a-list-item'}, text=True,
											 recursive=False):
					product_json['description'].append(spans.text.strip())

	except IOError as err:
		print("Error opening " + url)

	return product_json


if __name__ == "__main__":
	instances = []
	input_csv_path = sys.argv[1]
	output_csv_path = sys.argv[2]
	with open(input_csv_path) as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			instances.append(row)

	csv_columns = ['sample_id', 'user', 'query', 'product', 'attention_weight', 'drem_explanation', 'drem_attn_explanation', 'previous_reviews', 'title', 'image', 'description']

	# load all possible web pages
	product_set = set([instance['product'] for instance in instances])
	product_json_map = dict()
	for product in sorted(product_set):
		json_data = scrape("https://www.amazon.com/dp/" + product + "/", random.randint(0,100))
		if json_data and 'title' in json_data and json_data['title']:
			product_json_map[product] = json_data
			print('Success -> Scraped product : ' + product)
		else:
			print('Fail -> Scraped product : ' + product)

	with open(output_csv_path, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
		writer.writeheader()
		counter = 0
		for instance in instances:
			if instance['product'] in product_json_map:
				json_data = product_json_map[instance['product']]
			else:
				json_data = None

			if json_data:
				instance['title'] = json_data['title']
				instance['image'] =  json_data['image'] if 'image' in json_data else ""
				instance['description'] = json_data['description'] if 'description' in json_data else ""
				writer.writerow(instance)

			counter+=1
