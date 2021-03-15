from google.cloud import speech_v1p1beta1 as speech, storage
import io
import os

def download_from_yt(link, playlist=True):

    if playlist == True:
        command = "youtube-dl -x -o 'tmp/%(title)s.%(ext)s' --audio-format 'flac' --yes-playlist " + link
    else:
        command = "youtube-dl -x -o 'tmp/%(title)s.%(ext)s' --audio-format 'flac' " + link
    os.system(command)

    file = os.listdir('tmp/')[0]

    return file

def upload_to_gs(file, basedir):

    storage_client = storage.Client()

    bucket_list = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket(bucket_list[0].name)
    blob = bucket.blob(file)

    filename = basedir + '/tmp/' + file
    print('Uploading '+filename)
    blob.upload_from_filename(filename)
    print('Finished uploading '+filename)
    os.remove(filename)

    uri = 'gs://' + bucket_list[0].name + '/' + file

    return uri

def diarize(google_storage_uri):

    client = speech.SpeechClient()

    # Configuration
    enable_speaker_diarization = True
    diarization_speaker_count = 2
    language_code = "en-US"
    encoding = speech.enums.RecognitionConfig.AudioEncoding.FLAC
    config = {
        "language_code": language_code,
        "encoding": encoding,
        "enable_speaker_diarization": enable_speaker_diarization,
        "diarization_speaker_count": diarization_speaker_count,
        'audio_channel_count':2,
    }

    # Audio and text file
    audio = {'uri':google_storage_uri}
    filename = audio['uri']+'.txt'
    filename = 'speech2text/corpus/texts/'+filename.replace('gs://jrebot/', '')

    # Run diarizer
    print('Diarizing '+filename)
    operation = client.long_running_recognize(config, audio)
    response = operation.result()
    result = response.results[-1]
    words_info = result.alternatives[0].words

    # Format and output transcript
    tag = 1
    utterance = ""
    transcript = ''

    for word_info in words_info:
        if word_info.speaker_tag==tag:
            utterance = utterance+' '+word_info.word
        else:
            transcript += "speaker {}: {}".format(tag,utterance) + '\n'
            tag = word_info.speaker_tag
            utterance = ""+word_info.word

    #transcript += "speaker {}: {}".format(tag,speaker)
    with open(filename, 'w') as file:
        file.write(transcript)


def extract(transcript, speaker_number):

    with open(transcript,mode='r') as file:
         lines = file.readlines()

    filename = 'NLP/Texts/EX-'+transcript.replace('Speech2Text/Corpus/Texts/','')
    joe_speaker = 'speaker '+str(speaker_number)+':'
    with open(filename, mode='w') as file:
        for line in lines:
            if line.startswith(joe_speaker):
                num_words = len(line.split(' '))
                if num_words > 10:
                    file.write(line.replace(joe_speaker,''))

if __name__ == "__main__":
    extract('Speech2Text/Corpus/Texts/Joe Rogan Experience #1526 - Ali Macofsky.flac.txt',2)
    # 'gs://jrebot/Joe-Rogan-Experience-_1512-Ben-Shapiro-_64-kbps_.flac'
    # 'gs://jrebot/JRE_DO_test.flac'
