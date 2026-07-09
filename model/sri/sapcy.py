import spacy

nlp = spacy.load("en_core_web_sm")

resume = """
John Doe is a Python Developer.
He worked at Infosys for 2 years.
He completed B.E Computer Science.
He knows Python, SQL, Machine Learning.
"""

doc = nlp(resume)


print("Extracted Information:\n")

for ent in doc.ents:
    print(ent.text, " --> ", ent.label_)