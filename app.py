from flask import Flask, request, jsonify
import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Flask app
app = Flask(__name__)


# Load dataset
data_path = "iteung.chatgpt.csv"
df = pd.read_csv(data_path)

# Handle missing or non-string values
df['question'] = df['question'].fillna('').astype(str)

# Create LDA model
texts = [text.split() for text in df['question']]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p2')

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# Apply BERT embeddings to the question column
df['bert_embeddings'] = df['question'].apply(lambda x: get_bert_embeddings(x))

# Function to find the closest topic based on embeddings
def find_closest_topic(text, lda_model, dictionary):
    topic_probs = lda_model.get_document_topics(dictionary.doc2bow(text.split()))
    topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)
    return topic_probs[0][0]

# Map topics to embeddings
df['topic'] = df['question'].apply(lambda x: find_closest_topic(x, lda_model, dictionary))

# Function to calculate cosine similarity and find the best response
def get_best_response(user_input):
    user_embedding = get_bert_embeddings(user_input)
    if user_embedding.ndim == 1:
        user_embedding = user_embedding.reshape(1, -1)
    embeddings_list = list(df['bert_embeddings'].values)
    embeddings_array = np.vstack(embeddings_list)
    similarities = cosine_similarity(user_embedding, embeddings_array)
    best_match_idx = similarities.argmax()
    response = df['answer'].iloc[best_match_idx]
    # Remove newline characters
    response = response.replace('\n', ' ')
    return response

# Define API endpoint
@app.route('/webhook', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    response = get_best_response(user_input)
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
