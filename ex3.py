from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import read_data
from read_data import Readfiles

dataTrain = Readfiles('train.txt')
Train_X = read_data.word_tag_tuples()

