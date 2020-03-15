
import pickle
import os

class SimpleObj:
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
	os.system('pip install -r requirements.txt --no-index --find-links .')
