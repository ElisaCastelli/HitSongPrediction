import pandas as pd
import urllib.request as url_req
import progressbar
from time import sleep

# "/nas/home/ecastelli/thesis/"
df_url = pd.read_csv("../Data Sources/preview_url.csv")
bar = progressbar.ProgressBar(maxval=len(df_url), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


def download_single_song(track_id, url):
    path = "../Audio/" + track_id + ".mp3"
    url_req.urlretrieve(url, path)


def download_songs():
    bar.start()
    try:
        for index, row in df_url.iterrows():
            if row['preview_url'] is not None:
                print('Downloading and storing File: %s', index)
                download_single_song(str(row['track_id']), row['preview_url'])
                bar.update(index + 1)
                # path = "Audio/" + str(row['track_id']) + ".mp3"
                # url_req.urlretrieve(row['preview_url'], path)
        bar.finish()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    track_id = df_url.iloc[0]['track_id']
    url = df_url.iloc[0]['preview_url']
    download_single_song(track_id, url)
    # download_songs()
