import pandas as pd
import urllib.request as url_req
from time import sleep

# "/nas/home/ecastelli/thesis/"
df_url = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/bb_pop_url_DEF.csv")

def download_single_song(track_id, url):
    path = "/nas/home/ecastelli/thesis/Billboard/AudioBB/" + track_id + ".mp3"
    url_req.urlretrieve(url, path)


def download_songs():
    try:
        for index, row in df_url.iterrows():
            if row['preview_url'] is not None:
                print('Downloading and storing File:', index)
                download_single_song(str(row['spotify_id']), row['preview_url'])
    except Exception as e:
        print(e)


if __name__ == "__main__":
    download_songs()
