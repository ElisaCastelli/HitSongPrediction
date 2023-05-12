import os
from elasticsearch import Elasticsearch
import subprocess
import requests
from elasticsearch.helpers import bulk,scan
import urllib.request as url_req
import pandas as pd
#from audio_mgmt import AudioFeatures

class DataManager:
    def __init__(self):
        self.df_url = pd.read_csv("Data Sources/preview_url.csv", index_col=0)
        #self.config_file = config_file
        self.playlists = None
        self.country_playlists = None
        self.set_playlists_id = None
        #self.host = host
        #self.port = port
        self.audio_dir = os.getenv('Audio_dir')
        self.es = None
        #self.countries = countries
        #self.sp = SpotifyManager(self.config_file)
        #self.gen = GeniusManager(self.config_file)
        #self.connect_to_elasticsearch()
        #self.es_playlist_index = gv.indexNames

    def download_file(self):
            ok=False
            try:
                for index, row in self.df_url.head(100).iterrows():
                    if row['preview_url'] is not None :
                        #gv.logger.info('Downloading and storing File: %s', audio_name)
                        #request = 'curl ' + url + ' -o ' + audio_path 
                        #subprocess.call(request)
                        #response = requests.get(url)
                        #print(response)
                        path = "Audio/"+str(row['track_id'])+".mp3"
                        print(row['preview_url'])
                        print(path)
                        url_req.urlretrieve(row['preview_url'], path)
                        ok=True
            except Exception as e:
                 print(e)
                #gv.logger.error(e)
            return ok
    

d = DataManager()
d.download_file()
