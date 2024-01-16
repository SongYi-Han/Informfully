# Configuration settings for data enhancement pipeline

# Input and output file paths
input_file_path = './input/data.csv' # input file must be csv file 
output_file_path = './output/test.json' # output file must be json file

# Define attributes for enriching features
attributes = {'category' : {'enrich' : True, # Example: True or False
							'method' : 'metadata', # either 'metadata' or 'zero-shot'
							'meta_file_path' : './input/metadata.csv', # If not exists, leave as empty string (e.g. '')
							'candidate_labels': [] # Example: ["politics", "public health", "economics", "business", "sports", "life"]
							},  
			  'sentiment' : {'enrich' : True},
			  'complexity' : {'enrich' : True},
			  'named_entities' : {'enrich' : True,
			  					  'entities' : ['PERSON', 'GPE', 'ORG'], # Example: ['PERSON', 'GPE', 'ORG', 'NORP', 'FAC', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME']
			  					  'extend_ne_with_wiki': True,  # Example: True or False
			  					   },
			  'region' : {'enrich' : True},
			  'political_party' : {'enrich' : True},
			  'min_maj_ratio': {'enrich' : True, 
								'major_gender' : ['male'], # It can be multiple values such as ['male', 'female']
								'major_citizen': ['United States of America'], # It can be multiple values such as ['France', 'Italy']
								'major_ethinicity': ['white people'], # It can be multiple values such as ['white people', 'hispanic', 'asian']
								'major_place_of_birth': ['United States of America']},
			  'story': {'enrich' : True}} 
