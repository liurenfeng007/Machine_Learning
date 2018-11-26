import sys
sys.path.append('..')
import os
from numpy import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def one_hot(data):
	
	values = array(data)
	
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
	
	# binary encode
	onehot_encoder = OneHotEncoder(sparse = False)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	
	return onehot_encoded