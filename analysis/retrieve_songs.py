import pandas as pd
import urllib.request as url_req
csv_tracks = "./Data Sources/spotify_tracks.csv"
df_tracks = pd.read_csv(csv_tracks, index_col=0) 
df_url = df_tracks[['preview_url','uri']]
df_url['uri'] = df_url['uri'].apply(lambda x: str(x)[str(x).rfind(':')+1:])
df_url = df_url.rename({'uri': 'track_id'}, axis=1)
try:
    for index, row in df_url.iterrows():
        if row['preview_url'] is not None :
            print('Downloading and storing File: %s', index)
                        #request = 'curl ' + url + ' -o ' + audio_path 
                        #subprocess.call(request)
                        #response = requests.get(url)
                        #print(response)
            path = "Audio/"+str(row['track_id'])+".mp3"
                        #print(row['preview_url'])
                        #print(path)
            url_req.urlretrieve(row['preview_url'], path)
except Exception as e:
    print(e)