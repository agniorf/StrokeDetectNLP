from itertools import product
import argparse
import runLSTM_BOW_new as runLSTM_BOW

def main(batchSize, numEpoch, lstmSize, activationFunc, dropoutU, dropoutW, seed, job_id):
	i=0
	for item in product(batchSize, numEpoch, lstmSize, activationFunc, dropoutU, dropoutW, seed):
		if (i ==  int(job_id)-1):
			print('Argument:', job_id)
			print('Batch size:', item[0])
			print('Number of epochs:', item[1])
			print('Size of LSTM nodes:', item[2])
			print('Activation Function:', item[3])
			print('Dropout:', item[4])
			print('Recurrent Dropout:', item[5])
			print('Seed:', item[6])
			runLSTM_BOW.runSingle(item[0], item[1], item[2], item[3], item[4], item[5], item[6])
		i=i+1
	return i
