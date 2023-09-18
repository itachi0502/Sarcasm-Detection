from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings

def create_embeddings(embeddings_config):
    if 'TransformerWordEmbeddings' in embeddings_config:
        config = embeddings_config['TransformerWordEmbeddings']
        return TransformerWordEmbeddings(
            model=config['model'],
            layers=config['layers'],
            fine_tune=config['fine_tune'],
            pooling_operation=config['pooling_operation']
        )
    elif 'TransformerDocumentEmbeddings' in embeddings_config:
        config = embeddings_config['TransformerDocumentEmbeddings']
        return TransformerDocumentEmbeddings(
            model=config['model'],
            fine_tune=config['fine_tune']
        )
    else:
        raise NotImplementedError("Embeddings type not supported.")
