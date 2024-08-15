
import chromadb

class DatabaseAccessor():

    def __init__(self, collectionName, emb_fn):
        self.collectionName = collectionName
        self.emb_fn = emb_fn
    
    def connect(self):
        client = chromadb.HttpClient(host='localhost', port=8000)

        collection = client.get_collection(self.collectionName,embedding_function=self.emb_fn)
        return collection
        

