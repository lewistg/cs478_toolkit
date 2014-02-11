import numpy as np
import matplotlib.pyplot as plt
import csv

def parseCsvData(csvFileName):
	csvData = []
	with open(csvFileName, "rb") as csvfile:
		csvReader = csv.reader(csvfile, delimiter=",")
		for row in csvReader:
			csvData.append(row)
	return csvData

def plotAccAndMse():

	data = parseCsvData("iris.csv")

	tsAcc = [row[0] for row in data]
	tsAcc = tsAcc[1:]
	tsMse = [row[1] for row in data]
	tsMse = tsMse[1:]
	vsAcc = [row[2] for row in data]
	vsAcc = vsAcc[1:]
	vsMse = [row[3] for row in data]
	vsMse = vsMse[1:]

	assert(len(tsAcc) == len(tsMse))
	fig, tsAccPlot = plt.subplots()
	epochs = range(1, len(tsAcc) + 1)
	tsAccPlot.plot(epochs, tsAcc, "b-", label = "Accuracy for TS")
	tsAccPlot.plot(epochs, vsAcc, "g-", label = "Accuracy for VS")
	tsAccPlot.set_xlabel("epochs")
	tsAccPlot.set_ylabel("accuracy", color="b")
	for tick in tsAccPlot.get_yticklabels():
		tick.set_color("b")
	plt.legend(loc=0)

	tsMsePlot = tsAccPlot.twinx()
	tsMsePlot.plot(epochs, tsMse, "r.", label = "MSE for TS")
	tsMsePlot.plot(epochs, vsMse, "y.", label = "MSE for VS")
	tsMsePlot.set_ylabel("mean squared error", color="r")
	for tick in tsMsePlot.get_yticklabels():
		tick.set_color("r")

	plt.legend(loc=0)

	plt.show()
	

def main():
	plotAccAndMse()
	#data = parseCsvData("iris.csv")

if __name__ == "__main__":
	main()

