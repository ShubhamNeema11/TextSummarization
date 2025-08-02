    import re
    from nltk.corpus import stopwords
    from nltk.cluster.util import cosine_distance
    from nltk.tokenize import sent_tokenize
    import numpy as np
    import networkx as nx
    from flask import Flask, render_template, request

    def read_article(data):
        article = sent_tokenize(data)
        sentences = []
        for sentence in article:
            clean_sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
            words = clean_sentence.split(" ")
            sentences.append(words)
        return sentences

    def sentence_similarity(sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = set()
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
        all_words = list(set(sent1 + sent2))
        word_to_index = {word: i for i, word in enumerate(all_words)}
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        for w in sent1:
            if w not in stopwords and w in word_to_index:
                vector1[word_to_index[w]] += 1
        for w in sent2:
            if w not in stopwords and w in word_to_index:
                vector2[word_to_index[w]] += 1
        if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
            return 0.0
            
        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(sentences, stop_words):
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for idx1 in range(len(sentences)):
            for idx2 in range(idx1 + 1, len(sentences)):
                similarity = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
                similarity_matrix[idx1][idx2] = similarity
                similarity_matrix[idx2][idx1] = similarity
                
        return similarity_matrix

    def generate_summary(text, percentage=0.3):
        stop_words = stopwords.words('english')
        summarize_text = []
        original_sentences = sent_tokenize(text)
        sentences = read_article(text)

        if len(sentences) == 0:
            return "ERROR: No sentences found in the text."
        if len(sentences) < 2:
            return "ERROR: Too few sentences to meaningfully summarize. Please provide more text."

        top_n = max(1, int(len(sentences) * percentage))
        top_n = min(top_n, len(sentences))

        sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        ranked_sentence_with_index = sorted(((scores[i], i) for i, s in enumerate(sentences)), reverse=True)
        top_sentence_indices = [index for score, index in ranked_sentence_with_index[:top_n]]
        top_sentence_indices.sort()
        for index in top_sentence_indices:
            summarize_text.append(original_sentences[index])

        return " ".join(summarize_text)

    app = Flask(__name__)

    @app.route('/')
    def welcome():
        return render_template('index.html')

    @app.route('/submit', methods=['POST'])
    def submit():
        if request.method == 'POST':
            input_text = request.form['input']
            summary = generate_summary(input_text)
            return render_template('output.html', text=summary)
        return render_template('index.html')

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
