from data import Dataset, Labels
from utils import evaluate
import math
import os, sys


class Rocchio:
	def __init__(self):
		# centroids vectors for each Label in the training set.
		self.centroids = {l: {} for l in Labels}
		self.unique_words = {}

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Loop over all the samples in the training set, convert the
		documents to vectors and find the centroid for each Label.
		"""

		#get all the unique words
		words = set()
		for _, text, _ in ds: 
			text = text.split()
			words.update(text)
		words = sorted(words)
		self.unique_words = words

		#set all centroids to origin
		for label in Labels: 
			self.centroids[label] = {word: 0.0 for word in words}

		class_totals = {l: 0 for l in Labels}
		
		for id, text, label in ds: 
			class_totals[label] += 1
			doc_vec = {word: 0.0 for word in words}

			for word in text.split():
				doc_vec[word] += 1
			

			#normalize vector
			norm = math.sqrt(sum(count ** 2 for count in doc_vec.values()))
			doc_vec = {word: (doc_vec[word] / norm) for word in words}

			#add to centroids
			for word in words: 
				self.centroids[label][word] += doc_vec[word]
		#divide by class counts
		for label in Labels: 
			for word in words: 
				self.centroids[label][word] /= class_totals[label]

	def predict(self, x):
		"""
		x: string of words in the document.
		
		TODO: Convert x to vector, find the closest centroid and return the
		label corresponding to the closest centroid.
		"""

		vector = {word: 0.0 for word in self.unique_words}
		words = x.split()
		for word in words: 
			if word in vector: 
				vector[word] += 1


		scores = {l: 999999999 for l in Labels}


		for label in Labels: 
			centroid = self.centroids[label]
			scores[label] = math.sqrt(sum([(centroid[word] - vector[word])**2 for word in self.unique_words]))


		return min(scores, key=scores.get)

def main(train_split):
	rocchio = Rocchio()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	rocchio.train(ds)

	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(rocchio, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(rocchio, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(rocchio, test_ds)

if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
