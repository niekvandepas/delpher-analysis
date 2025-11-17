from gensim.models.fasttext import load_facebook_model

# path to downloaded Dutch fastText model (e.g., cc.nl.300.bin)
model_path = '/Users/niekvdpas/Downloads/cc.nl.300.bin'

# load model
print("loading model")
ft_model = load_facebook_model(model_path)

# get most similar words to key terms
query_terms = ['Indonesisch', 'Indo', 'rijsttafel', 'Javaanse']
for term in query_terms:
    try:
        sims = ft_model.wv.most_similar(term, topn=20)
        print(f"Top neighbours for {term}:")
        for word, score in sims:
            print(f"  {word} {score:.3f}")
    except KeyError:
        print(f"Term '{term}' not in vocabulary")
