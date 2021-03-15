import tensorflow_cloud as tfc
import os
import nbconvert

basedir = '/Users/armandrego/Documents/Python/JREBot/'
credentials_json = basedir + 'Speech2Text/creds.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_json

GCP_BUCKET = 'jrebot'

tfc.run(entry_point='NLP/bot_model-2.py',
requirements_txt="NLP/requirements.txt",
docker_image_bucket_name=GCP_BUCKET,
stream_logs=True)
