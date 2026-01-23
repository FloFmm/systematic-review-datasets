import numpy as np
from csmed.experiments.measures import evaluate_model_by_positive_count
from csmed.experiments.csmed_cochrane_retrieval import load_dataset
SEED = 42
QUERY_TYPES =  ["title_abstract"]#, "title", "abstract"]#, "query", "criteria"]
TOTAL_DOCS = 503679
if __name__ == "__main__":
    np.random.seed(SEED)
    retriever_configs = {
        # "bm25": {
        #     "type": "sparse",
        #     "model": "bm25",
        # },
        # "tf-idf": {
        #     "type": "sparse",
        #     "model": "tf-idf",
        # },
        # "MedCPT": {
        #     "type": "dense",
        #     "model": "ncbi/MedCPT-Article-Encoder",
        #     "query_model": "ncbi/MedCPT-Query-Encoder",
        #     "max_length": 256,
        # },
        # "MedCPT-Doc-Enc-Only": {
        #     "type": "dense",
        #     "model": "ncbi/MedCPT-Article-Encoder",
        #     "max_length": 256,
        # },
        # "MiniLM-128": {
        #     "type": "dense",
        #     "model": "sentence-transformers/all-MiniLM-L6-v2",
        #     "max_length": 128,
        # },
        # "MiniLM-256": {
        #     "type": "dense",
        #     "model": "sentence-transformers/all-MiniLM-L6-v2",
        #     "max_length": 256,
        # },
        # "qa-MiniLM-512": {
        #     "type": "dense",
        #     "model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        #     "max_length": 512,
        # },
        # "mpnet": {
        #     "type": "dense",
        #     "model": "sentence-transformers/all-mpnet-base-v2",
        #     "max_length": 512,
        # },
        # "nli-mpnet": {
        #     "type": "dense",
        #     "model": "sentence-transformers/nli-mpnet-base-v2",
        #     "max_length": 512,
        # },
        # "biobert-nli": {
        #     "type": "dense",
        #     "model": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        #     "max_length": 512,
        # },
        # "S-BioBert": {
        #     "type": "dense",
        #     "model": "pritamdeka/S-BioBert-snli-multinli-stsb",
        #     "max_length": 512,
        # },
        "pubmedbert": {
            "type": "dense",
            "model": "pritamdeka/S-PubMedBert-MS-MARCO",
            "max_length": 512,
        },
        # "roberta": {
        #     "type": "dense",
        #     "model": "sentence-transformers/stsb-roberta-base-v2",
        #     "max_length": 512,
        # },
    }

    dataset = load_dataset()
    # mini dataset
    # max_items = 20
    # mini_dataset = {"EVAL":{}}
    # count = 0
    # for review_name, review_data in dataset["EVAL"].items():
    #     mini_dataset["EVAL"][review_name] = review_data
    #     count += 1
    #     if count >= max_items:
    #         break

    # dataset = mini_dataset
    # mini dataset
    eval_reviews = dataset["EVAL"] | dataset["TRAIN"]
    
    total_docs = TOTAL_DOCS
    for split, reviews in dataset.items():
        print(f"\n=== Split: {split} ===")
        print(f"Number of reviews: {len(reviews)}")
    print("total_docs:", total_docs)
    for name, conf in retriever_configs.items():
        print("Processing model:", name)

        qrels_dict = {}
        
        for index, (review_name, review_data) in enumerate(eval_reviews.items(), start=1):
            qrels = {
                doc["pmid"]: int(doc["label"])
                for doc in review_data["data"]["train"]
            }

            qrels_dict[review_name] = qrels

        for query_type in QUERY_TYPES:
            evaluate_model_by_positive_count(
                model_name=name,
                query_type=query_type,
                total_docs=total_docs,
                qrels_dict=qrels_dict,
                output_dir = "../boolean-query-generation/data/reports/title_and_abstract",
                rankings_base_path = "../systematic-review-datasets/data/rankings"
            )