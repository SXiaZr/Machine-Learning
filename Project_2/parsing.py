import numpy as np
import pandas as pd
import re

from bs4 import BeautifulSoup

def write_txt_file(results):

	file = open ("to_parse.txt","w")
	file.write(results.records)
	file.close()
	return file

def  read_txt_file_to_xml(file):

	file_object = open("to_parse.txt")
	to_be_parsed = file.file_object.read()
	Soup = BeautifulSoup(to_be_parsed,'lxml')
	return Soup

def construct_dataset(Soup):
	#pd.set_option('max_colwidth',5000)
	paper_list = Soup.find_all('rec')
	data = []

	for paper in paper_list:

		title = paper.find('title',type='item').text
		uid = paper.find('uid').text
		publish_date = paper.find('pub_info').get('sortdate')
		vol = paper.find('pub_info').get('vol')
		pubtype = paper.find('pub_info').get('pubtype')
		issue = paper.find('pub_info').get('issue') 
		language = list(map(lambda x: x.text, paper.find('languages').find_all('language', type='primary')))
		doctype = paper.find('doctype').text
		source = paper.find('title', type='source').text
		keywords = list(map(lambda x: x.text, paper.find_all('keyword')))
		abstract = paper.find('p').text if paper.find('p')!=None else paper.find('p')
		headings = list(map(lambda x: x.text, paper.find_all('heading'))) \
				if len(paper.find_all('heading'))!=0 else None
		subheadings = list(map(lambda x: x.text, paper.find_all('subheading'))) \
					if len(paper.find_all('subheading'))!=0 else None
		traditional_subjects = list(map(lambda x: x.text, paper.find_all('subject', ascatype="traditional"))) \
							if len(paper.find_all('subject', ascatype="traditional"))!=0 else None
		extended_subjects = list(map(lambda x: x.text, paper.find_all('subject', ascatype="extended"))) \
						if len(paper.find_all('subject', ascatype="extended"))!=0 else None
		category_info = {'headings':headings, 'subheadings':subheadings, 
					 'traditional_subjects':traditional_subjects, 'extended_subjects':extended_subjects}
	
		# Address info
		addresses = paper.find(lambda tag: tag.has_attr('count') and tag.name=='addresses')
		addresses_list = addresses.find_all('address_spec')
		addresses_info_dict = {}

		for address in addresses_list:
			addr_no = address.get('addr_no')
			full_address = address.find('full_address').text if address.find('full_address') else np.nan
			organization = address.find('organization').text if address.find('organization') else np.nan
			city = address.find('city').text
			country = address.find('country').text
			addresses_info_dict.update({'addr_no_' + addr_no: {'full_address':full_address,\
														   'organization': organization,\
														   'city':city,\
														   'country':country}})
	#print(addresses_info_dict)
	
		# Author information
		names = paper.find(lambda tag: tag.has_attr('count') and tag.name=='names')
		names_list = names.find_all('name')
		name_info_dict = {}
		for name in names_list:
			# Basic name info
			daisng_id = name.get('daisng_id') 
			role = name.get('role') 
			seq_no = name.get('seq_no') 
			display_name = name.find('display_name').text
			full_name = name.find('full_name').text
			# Zip as a dictionary and add to list
			name_info_personal = {'daisng_id':daisng_id,\
							  'role':role,\
							  'seq_no':seq_no,\
							  'display_name': display_name}

		#print(name_info_dict)
		# If the relationship between address and name exist
			if name.get('addr_no') != None:
				name_addr_no = name.get('addr_no')
				if len(name_addr_no) > 1 :
					name_addr_no = name_addr_no[0]
					#print(addresses_info_dict['addr_no_'+name_addr_no])
				name_info_personal.update({'address':addresses_info_dict['addr_no_' + name_addr_no]})
			else:
				name_info_personal.update(addresses_info_dict)
			name_info_dict.update({full_name:name_info_personal})

		data.append({'title': title, 
				 'uid': uid, 
				 'publish_date': publish_date,
				 'vol':vol,
				 'pubtype':pubtype,
				 'issue':issue,
				 'language':language,
				 'doctype':doctype,
				 'source':source,
				 'name_info':name_info_dict,
				 'keywords':keywords,
				 'category_info':category_info,
				 'abstract':abstract})
	publications = pd.DataFrame.from_dict(data)

	return publications

						  

