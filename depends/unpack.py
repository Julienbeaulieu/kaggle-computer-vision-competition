
import pickle
import os

class SimpleObj:
	"""
	This class data structure is required to decompress the entry and bin data.
	"""
	def __init__(self,fname, bdata):
		self.name = fname
		self.bdata= bdata
		return

if __name__ == '__main__':

	picklefile = 'dill.pkl'

	# unpack pickle
	installers = []
	with open(picklefile, 'rb') as pf:
		installers = pickle.load(pf)

	for i in installers:
		with open(i.name, 'wb') as p:
			p.write(i.bdata)

	# install
	# If all goes well, it should return 0
	os.system('pip install -r requirements.txt --no-index --find-links .')

