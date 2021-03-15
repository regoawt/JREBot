import jrebot
import os

basedir = '/Users/armandrego/Documents/Python/JREBot/'
credentials_json = basedir + 'Speech2Text/creds.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_json

link_list = ['https://www.youtube.com/playlist?list=PLtwBKErCKOdkyVJy-Ag4prCKNp7mddkxc']

uri_list = []

# for link in link_list:
#     file = jrebot.download_from_yt(link, playlist=True)

for file in os.listdir('tmp/'):
    uri = jrebot.upload_to_gs(file, basedir)

    uri_list.append(uri)

with open('uri_list.txt','w') as uri_file:
    uri_file.write(str(uri_list))

for uri in uri_list:
    jrebot.diarize(uri)
