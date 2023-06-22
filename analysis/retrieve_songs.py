import pandas as pd
import urllib.request as url_req

df_url = pd.read_csv("./Data Sources/preview_url.csv")


def download_songs():
    try:
        for index, row in df_url.iterrows():
            if row['preview_url'] is not None:
                print('Downloading and storing File: %s', index)
                path = "Audio/" + str(row['track_id']) + ".mp3"
                url_req.urlretrieve(row['preview_url'], path)
    except Exception as e:
        print(e)

# TODO download songs on server: try with one and then download all the songs during the night, maybe you can use curie
