from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

resume = "I have skills in Python, SQL and Machine Learning."

job = "Looking for a Python Developer with SQL knowledge."

resume_embedding = model.encode([resume])

job_embedding = model.encode([job])

score = cosine_similarity(resume_embedding, job_embedding)

print("Similarity Score:", score[0][0])