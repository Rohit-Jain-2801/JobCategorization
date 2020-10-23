print('Please Wait...')

# Importing libraries 
import imaplib, email, getpass
import jobcategorizer

user = input('Enter Email-id: ')
password = getpass.getpass('Enter Password: ')
imap_url = 'imap.gmail.com'

# Function to search for a key value pair 
def search(key, value, con): 
	result, data = con.search(None, key, '"{}"'.format(value)) 
	return data 

# Function to get the list of emails under this label 
def get_emails(result_bytes): 
	msgs = [] # all the email data are pushed inside an array 
	for num in result_bytes[0].split(): 
		typ, data = con.fetch(num, '(RFC822)') 
		msgs.append(data) 
	return msgs 

# this is done to make SSL connnection with GMAIL 
con = imaplib.IMAP4_SSL(imap_url) 

# logging the user in 
con.login(user, password) 

# calling function to check for email under this label 
con.select('Inbox') 

# fetching emails from this user "tu**h*****1@gmail.com" 
msgs = get_emails(search('SUBJECT', 'Thank you for applying', con)) 

# printing them by the order they are displayed in your gmail 
for msg in msgs[::-1]:
	for sent in msg: 
		if type(sent) is tuple: 

			# encoding set as utf-8 
			content = str(sent[1], 'utf-8')
			data = str(content)

			# Handling errors related to unicodenecode 
			try: 
				# indexstart = data.find("ltr")
				search_data = '<div dir="auto">'
				indexstart = data.find(search_data)
				data2 = data[indexstart + len(search_data): len(data)]
				indexend = data2.find("</div>") 

				# printtng the required content which we need 
				# to extract from our email i.e our body 
				jobcategorizer.load_models()
				role, category = jobcategorizer.find_category(txt=data2[0: indexend])
				if role:
					print(role, '-->', category)
				else:
					print(category)

			except UnicodeEncodeError as e: 
				pass