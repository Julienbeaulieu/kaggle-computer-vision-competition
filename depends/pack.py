
import pickle
import os
from os import path

class SimpleObj:
	"""
	This class data structure is required to store the entry and bin data.
	"""
	def __init__(self,fname, bdata):
		self.name = fname
		self.bdata= bdata
		return

if __name__ == '__main__':

	reqs = 'requirements.txt'
	picklefile = 'dill.pkl'
	basedir = 'wheelhouse'
	installers = []


	if not path.exists(basedir):
		os.mkdir(basedir)

	# download requirements to basedir
	os.system('pip download -r requirements.txt -d wheelhouse')


	# pack into pickle
	for entry in os.listdir(basedir):
		filename = os.path.join(basedir, entry)
		if os.path.isfile(filename):
			print(filename)
			with open(filename, 'rb') as fl:
				allBinData = fl.read()
				d = SimpleObj(entry, allBinData)

				installers.append(d)
			
	with open(picklefile, 'wb') as pf:
		pickle.dump(installers, pf)
		pf.flush()
