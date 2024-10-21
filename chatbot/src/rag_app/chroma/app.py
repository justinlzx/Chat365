from .create_database import process_documents


DATA_PATHS = ["https://www.healthhub.sg/programmes/nsc"]  # add more sites as required

embeddings = process_documents(DATA_PATHS)
