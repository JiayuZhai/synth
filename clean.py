# clean.py
import csv
import numpy as np

def convert(x):
    if x == 'NA':
        x = -1
    return float(x)

def main():
	# variables we need
	required = [2,4,5,6,7,11,12,13,15,23]
	# continuous variables
	var_con = [5,14,15,16,17,18,19,20,21,22,23,24]
	with open('data/challenge.csv','r') as file:
	    x = csv.reader(file)
	    # log projection between categories and its number
	    var_distinct = [{} for i in range(25)]
	    table = np.zeros((82851,25))
	    for i,z in enumerate(x):
	        if i==0:
	            var_names = np.array(z)
	        if i!=0:
	            record = np.array(z)
	            for j in range(25):
	                if j in var_con:
	                    record[j] = convert(record[j])
	                    continue
	                if record[j] not in var_distinct[j]:
	                    var_distinct[j][(record[j])] = len(var_distinct[j])
	                record[j] = var_distinct[j][(record[j])]
	            table[i-1,:] = record

	# save clean data for what we required
	print(var_names[required])
	np.save('data/clean_data.npy',table[:,required])
	print('clean data saved')
	np.save('data/categories.npy',var_distinct)

	# test the clean original data
	# tablenew = np.load('data/clean_data.npy')
	# print(tablenew[0:10,:])

if __name__ == '__main__':
	main()