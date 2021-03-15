import jrebot
from google.cloud import storage
import os

basedir = '/Users/armandrego/Documents/Python/JREBot/'
credentials_json = basedir + 'Speech2Text/creds.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_json

storage_client = storage.Client()

bucket_list = list(storage_client.list_buckets())
bucket = storage_client.get_bucket(bucket_list[0].name)
blobs = bucket.list_blobs()

for blob in blobs:
  name = blob.name
  uri = 'gs://' + bucket_list[0].name + '/' + name
  print(uri)
  jrebot.diarize(uri)
