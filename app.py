from flask import Flask, request, jsonify
import pandas as pd
import requests
import json
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Kredensial WhatsApp Cloud API
WHATSAPP_API_URL = 'https://graph.facebook.com/v20.0/357553994118945/messages'
ACCESS_TOKEN = 'EAAHAiuf8TigBO8YfvfsSLErsZAI15l0pknEx4G9anU5HohJfWndD0CL44685umCksjrdvEioZCxI03JZBAo0Vhhs1njwL6QourZCUEJ9g0HQpMpZApwjZA0ZCIYZBV4F2bRUatZCgK68JUcGR3yA9nDub8sbfL4LCQHcQpF0nRwg3UxgnbB7gbcYtLV1Cix88QJNqopFJ9dyopYrUNabfX0AZD'
VERIFY_TOKEN = '123456'  # Token verifikasi yang Anda tetapkan di Meta

# Muat dataset
data_path = "iteung.chatgpt.csv"
df = pd.read_csv(data_path)

# Tangani nilai yang hilang atau bukan string
df['question'] = df['question'].fillna('').astype(str)

# Buat model LDA
texts = [text.split() for text in df['question']]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

# Muat tokenizer dan model BERT
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p2')

# Fungsi untuk mendapatkan embedding BERT
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# Terapkan embedding BERT ke kolom pertanyaan
df['bert_embeddings'] = df['question'].apply(lambda x: get_bert_embeddings(x))

# Fungsi untuk menemukan topik terdekat berdasarkan embedding
def find_closest_topic(text, lda_model, dictionary):
    topic_probs = lda_model.get_document_topics(dictionary.doc2bow(text.split()))
    topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)
    return topic_probs[0][0]

# Pemetaan topik ke embedding
df['topic'] = df['question'].apply(lambda x: find_closest_topic(x, lda_model, dictionary))

# Fungsi untuk menghitung kemiripan kosinus dan menemukan respons terbaik
def get_best_response(user_input):
    user_embedding = get_bert_embeddings(user_input)
    if user_embedding.ndim == 1:
        user_embedding = user_embedding.reshape(1, -1)
    embeddings_list = list(df['bert_embeddings'].values)
    embeddings_array = np.vstack(embeddings_list)
    similarities = cosine_similarity(user_embedding, embeddings_array)
    best_match_idx = similarities.argmax()
    response = df['answer'].iloc[best_match_idx]
    response = response.replace('\n', ' ')
    return response

# Endpoint untuk verifikasi webhook
@app.route('/webhook', methods=['GET'])
def verify_webhook():
    if request.args.get('hub.mode') == 'subscribe' and request.args.get('hub.challenge'):
        if request.args.get('hub.verify_token') == VERIFY_TOKEN:
            return request.args['hub.challenge'], 200
        return 'Verification token mismatch', 403
    return 'Hello World', 200

# Endpoint untuk menerima pesan dari WhatsApp
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    from_number = data['entry'][0]['changes'][0]['value']['messages'][0]['from']
    incoming_msg = data['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
    
    # Dapatkan respons terbaik
    response_message = get_best_response(incoming_msg)
    
    # Kirim respons melalui WhatsApp Cloud API
    payload = {
        'messaging_product': 'whatsapp',
        'to': from_number,
        'text': {'body': response_message}
    }
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    response = requests.post(WHATSAPP_API_URL, headers=headers, data=json.dumps(payload))
    
    return jsonify({'status': 'Pesan terkirim'})

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
