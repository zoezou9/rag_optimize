import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from mabwiser.mab import MAB, LearningPolicy
from transformers import pipeline

from datasets import load_dataset

# Natural Questions, SQuAD, and FEVER datasets
nq_dataset = load_dataset("natural_questions", split="train")
squad_dataset = load_dataset("squad", split="train")
fever_dataset = load_dataset("fever", split="train")

nq_questions = [item["question"] for item in nq_dataset]
squad_questions = [item["question"] for item in squad_dataset]
fever_claims = [item["claim"] for item in fever_dataset]  # Treat claims as queries

nq_passages = [item["document_text"] for item in nq_dataset]
squad_contexts = [item["context"] for item in squad_dataset]
fever_evidence = [item["evidence"] for item in fever_dataset]


# SBERT for Dense Retrieval
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample
# documents = ["The capital of France is Paris.", "Quantum mechanics describes the behavior of particles.", "Neural networks are used in deep learning."]
# tokenized_docs = [doc.split() for doc in documents]

# Combine all passages and queries
documents = nq_passages + squad_contexts + fever_evidence
queries = nq_questions + squad_questions + fever_claims
tokenized_docs = [doc.split() for doc in documents]

# BM25 Indexing
bm25 = BM25Okapi(tokenized_docs)

# Dense Retrieval Indexing with FAISS
doc_embeddings = sbert_model.encode(documents, convert_to_tensor=True)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings.numpy())

# Define Bandit for Optimization
bandit = MAB(arms=[(3, 0.5, 0.5), (5, 0.6, 0.4), (7, 0.7, 0.3)], learning_policy=LearningPolicy.LinUCB(alpha=1.0))
bandit.fit([], [])  # Initial training with no data

# Function to Retrieve Documents
def retrieve_documents(query, top_k=5, lambda_weight=0.5):
    # BM25 Retrieval
    bm25_scores = bm25.get_scores(query.split())
    top_bm25 = np.argsort(bm25_scores)[-top_k:]
    
    # Dense Retrieval
    query_embedding = sbert_model.encode([query], convert_to_tensor=True)
    _, top_faiss = index.search(query_embedding.numpy(), top_k)
    
    # Hybrid Score Combination
    final_scores = lambda_weight * np.array([bm25_scores[i] for i in top_bm25]) + \
                   (1 - lambda_weight) * np.array([doc_embeddings[i] @ query_embedding.T for i in top_faiss[0]])
    
    return [documents[i] for i in np.argsort(final_scores)[-top_k:]]

# Define Reward Function
def compute_reward(retrieved_docs, ground_truth):
    retrieval_accuracy = len(set(retrieved_docs) & set(ground_truth)) / len(ground_truth)
    token_cost = sum(len(doc.split()) for doc in retrieved_docs) / 1000  # Normalized cost
    return retrieval_accuracy - token_cost

# # Simulate Adaptive Retrieval
# query = "What is the capital of France?"
# ground_truth = ["The capital of France is Paris."]

# # Select Action from Bandit
# top_k, threshold, lambda_weight = bandit.predict([query])
# retrieved_docs = retrieve_documents(query, top_k, lambda_weight)
# reward = compute_reward(retrieved_docs, ground_truth)

# # Update Bandit with Reward
# bandit.partial_fit([query], [reward])

# print("Retrieved Documents:", retrieved_docs)
# print("Reward:", reward)

# Simulate Adaptive Retrieval with Dataset Queries
for query in queries[:10]:  # Test with first 10 queries
    ground_truth = [query]  # Simplification for testing
    
    # Select Action from Bandit
    top_k, threshold, lambda_weight = bandit.predict([query])
    retrieved_docs = retrieve_documents(query, top_k, lambda_weight)
    reward = compute_reward(retrieved_docs, ground_truth)
    
    # Update Bandit with Reward
    bandit.partial_fit([query], [reward])
    
    print(f"Query: {query}")
    print("Retrieved Documents:", retrieved_docs)
    print("Reward:", reward)
    print("-" * 50)
