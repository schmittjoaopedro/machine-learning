from transformers import pipeline

classifier = pipeline('sentiment-analysis')
resp = classifier('We are very happy to introduce pipeline to the transformers repository.')
print(resp)
