from data import Dataset, Labels
from utils import evaluate
import os, sys
import math
from collections import Counter

K = 5

class KNN:
	def __init__(self):
		# bag of words document vectors
		self.bow = []
		self.unique_words = {}

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Save all the documents in the train dataset (ds) in self.bow.
		You need to transform the documents into vector space before saving
		in self.bow.
		"""
		words = set()
		for _, text, _ in ds: 
			text = text.split()
			words.update(text)
		words = sorted(words)
		self.unique_words = words
		#i = 0
		for _, text, label in ds: 
			#i += 1
			#print(f"progress: {i} / {len(ds)}")
			vector = {word: 0 for word in words}
			for word in text.split():
				vector[word] += 1
			self.bow.append((label, vector))

	def predict(self, x):
		"""
		x: string of words in the document.

		TODO: Predict class for x.
		1. Transform x to vector space.
		2. Find k nearest neighbors.
		3. Return the class which is most common in the neighbors.
		"""
		#print("predict")

		#print("processing vector")
		doc_vector = {word: 0 for word in self.unique_words}
		for word in x.split():
			if word in doc_vector: 
				doc_vector[word] += 1
		doc_magn = math.sqrt(sum(v**2 for v in doc_vector.values()))

		distances = []

		#print("iterating through vectors!")
		for label, vector in self.bow: 
			dot_product = sum([(vector[word] * doc_vector[word]) for word in self.unique_words])
			magn = math.sqrt(sum(v ** 2 for v in vector.values()))
			cos_sin = dot_product / (magn * doc_magn)

			distances.append((label, cos_sin))
		#print("sorting ... ")
		distances.sort(key=lambda x: x[1], reverse=True)
		top_k_labels = [label for label, _ in distances[:K]]
		return Counter(top_k_labels).most_common(1)[0][0]

def main(train_split):
	knn = KNN()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	knn.train(ds)

	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(knn, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(knn, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(knn, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
