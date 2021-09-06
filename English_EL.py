import requests
from http.server import HTTPServer

from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner
from REL.server import make_handler
from REL.entity_disambiguation import EntityDisambiguation


wiki_version = "wiki_2019"
base_url = "home/yuqi/ex_sum/EL_data/"

config = {
    "mode": "eval",
    "model_path": base_url+"ed-wiki-2019",
}

model = EntityDisambiguation(base_url, wiki_version, config)

# Using Flair:
tagger_ner = load_flair_ner("ner-fast")

# Alternatively, using n-grams:
tagger_ngram = Cmns(base_url, wiki_version, n=5)

server_address = ("127.0.0.1", 1235)
server = HTTPServer(
    server_address,
    make_handler(
        base_url, wiki_version, model, tagger_ner
    ),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)

IP_ADDRESS = "http://localhost"
PORT = "1235"
text_doc = "Apple is an company."

document = {
    "text": text_doc,
    "spans": [],  # in case of ED only, this can also be left out when using the API
}

API_result = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document).json()

print(API_result)
exit(0)

