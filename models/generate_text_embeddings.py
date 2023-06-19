import json, os, copy, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pandas as pd

df_songs = pd.read_csv("Data Sources/spotify_tracks.csv")

def mean_pooling(model_output, attention_mask):
    """
    Pool the vectors in a series of vectors into one vector using
        the model's attention values to weight the individual token vectors appropriately
        
    Code from https://www.sbert.net/examples/applications/computing-embeddings/README.html
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    ### Modified from documentation so function returns a list of values from tensor
    ### Lists are easier than tensors or arrays to read from / write to json
    return (sum_embeddings / sum_mask).numpy().tolist()


def gen_embedding(text, model, tokenizer):
    """
    Take in a text, tokenize and encode it using the sBERT tokenizer and model respectively.

    Pool the outputs into a single vector using the attention mask from the tokenizer.

    Code from https://medium.com/swlh/transformer-based-sentence-embeddings-cd0935b3b1e0
    """
    ### Tokenize the texts
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    ### Encode the tokenized data with model
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    ### Pool the outputs into a single vector
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


def add_embeddings(data, min_songs=8):
    """
    Given data scraped from genius, generate and add embeddings of each songs' lyrics

    Lyrics are tokenized individually, the series of tokens is fed to the model. The output
        of the model is then combined with the attention mask to create a single, 768d vector
        for each song.
    """
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    ### Create copy of the data so we can update elements
    #data_loop = copy.deepcopy(data)

    ### Filter out any albums with <= 7 songs
    #data_loop = {artist: {album:songs for album, songs in albums.items() if \
    #                        len(songs) >= min_songs} \
    #                for artist,albums in data_loop.items()}
#
    #for artist, albums in data_loop.items():
    #    for album, songs in tqdm(albums.items(), desc=f'{artist} albums'):
    #        for song in songs:
    #            song.update({'embedding': gen_embedding(song.get('lyrics'), model, tokenizer)})
    #            
    #return data_loop
    embeddings = gen_embedding(data, model, tokenizer)
    return (f"Embedding: {embeddings}, size: {len(embeddings[0])}")
    

def main():
    """
    Download scraped files with lyrics, generate embeddings, upload embeddings to GCS
    """
    
    #download_blob('data/artist_albums_lyrics_0607.json', '/tmp/artist_albums_lyrics_0607.json')
#
    #with open('/tmp/artist_albums_lyrics_0607.json', 'r') as f:
    #    data = json.load(f)
#
    #data_emb = add_embeddings(data)
#
    #with open('artist_albums_lyrics_embs_0608.json', 'w') as f:
    #    json.dump(data_emb, f, indent=4)
#
    #upload_blob('artist_albums_lyrics_embs_0608.json', folder_name='data')
    lyrics_prova = df_songs.iloc[12787]['lyrics']
    id_prova = df_songs.iloc[12787]['uri']
    print(add_embeddings(lyrics_prova))
    

if __name__ == "__main__":
    main()