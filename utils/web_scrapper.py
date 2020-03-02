import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import ssl
import csv
import sys


def scrape(url):
	# For ignoring SSL certificate errors
	ctx = ssl.create_default_context()
	ctx.check_hostname = False
	ctx.verify_mode = ssl.CERT_NONE

	product_json = dict()

	# user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
	# values = {'name': 'Michael Foord',
	# 		  'location': 'Northampton',
	# 		  'language': 'Python'}
	# headers = {'User-Agent': user_agent}
	#
	# data = urllib.parse.urlencode(values)
	# data = data.encode('ascii')
	# req = urllib.request.Request(url, data, headers)

	try:
		html = urllib.request.urlopen(url, context=ctx).read()
		soup = BeautifulSoup(html, 'html.parser')

		# This block of code will help extract the Prodcut Title of the item

		for spans in soup.findAll('span', attrs={'id': 'productTitle'}):
			name_of_product = spans.text.strip()
			product_json['title'] = name_of_product
			break

		# This block of code will help extract the image of the item

		for divs in soup.findAll('div', attrs={'id': 'rwImages_hidden'}):
			for img_tag in divs.findAll('img', attrs={'style': 'display:none;'
													  }):
				product_json['image'] = img_tag['src']
				break

		# This block of code will help extract top specifications and details of the product

		product_json['description'] = []
		for ul_tags in soup.findAll('ul',
									attrs={'class': 'a-unordered-list a-vertical a-spacing-none'
										   }):
			for li_tags in ul_tags.findAll('li'):
				for spans in li_tags.findAll('span',
											 attrs={'class': 'a-list-item'}, text=True,
											 recursive=False):
					product_json['description'].append(spans.text.strip())

	except IOError:
		print("Error opening " + url)

	return product_json


if __name__ == "__main__":
	instances = []
	input_csv_path = sys.argv[1]
	with open(input_csv_path) as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			instances.append(row)

	csv_columns = ['user', 'query', 'product', 'explanation', 'previous_reviews', 'title', 'image', 'description']


	with open('mturk-batch-input.csv', 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
		writer.writeheader()
		product_json_map = dict()

		for instance in instances:
			if instance['product'] not in product_json_map:
				json_data = scrape("https://www.amazon.com/dp/" + instance['product'] + "/")
				product_json_map[instance['product']] = json_data
			else:
				json_data = product_json_map[instance['product']]

			if json_data and json_data['title']:
				instance['title'] = json_data['title']
				instance['image'] =  json_data['image'] if 'image' in json_data else ""
				instance['description'] = json_data['description'] if 'description' in json_data else ""
				writer.writerow(instance)
