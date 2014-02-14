import csv
import matplotlib.pyplot as plt
import numpy as np

def parseCsvData(csvFileName):
	csvData = []
	with open(csvFileName, "rb") as csvfile:
		csvReader = csv.reader(csvfile, delimiter=",")
		for row in csvReader:
			csvData.append(row)
	return csvData


def plotAccAndMse(csvFileName):
	data = parseCsvData(csvFileName)

	learningRate = [row[0] for row in data]
	learningRate = np.array(learningRate[1:], dtype=float)
	tsAcc = [row[1] for row in data]
	tsAcc = tsAcc[1:]
	tsMse = [row[2] for row in data]
	tsMse = tsMse[1:]
	vsAcc = [row[3] for row in data]
	vsAcc = vsAcc[1:]
	vsMse = [row[4] for row in data]
	vsMse = vsMse[1:]

	assert(len(tsAcc) == len(tsMse))
	fig, tsAccPlot = plt.subplots()
	tsAccPlot.plot(learningRate, tsAcc, "b-", label = "Accuracy for TS")
	tsAccPlot.plot(learningRate, vsAcc, "g-", label = "Accuracy for VS")
	tsAccPlot.set_xlabel("Learning Rate", fontsize=18)
	tsAccPlot.set_ylabel("Accuracy", fontsize=18, color="b")
	for tick in tsAccPlot.get_yticklabels():
		tick.set_color("b")
	yin, yax = plt.ylim()
	plt.ylim(ymax = yax + .2)
	plt.legend(loc=2)
	plt.title("Accuracy and MSE vs. Learning Rate", fontsize=24)

	tsMsePlot = tsAccPlot.twinx()
	tsMsePlot.plot(learningRate, tsMse, "r", linestyle="--", label = "MSE for TS")
	tsMsePlot.plot(learningRate, vsMse, "m", linestyle="--", label = "MSE for VS")
	yin, yax = plt.ylim()
	plt.ylim(ymax = yax + .1)
	tsMsePlot.set_ylabel("Mean Squared Error (MSE)", color="r", fontsize=18)
	for tick in tsMsePlot.get_yticklabels():
		tick.set_color("r")

	plt.legend(loc=1)

	plt.show()

def calcData():
	print "LR, TS ACC, TS MSE, VS ACC, VS MSE"
	minVsMse = 1000
	minVsMseLr = 0
	for lr in ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.2",\
		"0.21", "0.22", "0.23", "0.24", "0.25", "0.26", "0.27", "0.28", "0.29", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7"]:
		tsAcc = 0.0
		tsMse = 0.0
		vsAcc = 0.0
		vsMse = 0.0
		for i in range(5):
			data = parseCsvData(lr + "_trial%d.txt" % i)

			# find the record for 100 before the end
			data = np.array(data[1:], dtype=float)
			bestRecord = len(data) - 102
			tsAcc += data[bestRecord][0]
			tsMse += data[bestRecord][1]
			vsAcc += data[bestRecord][2]
			vsMse += data[bestRecord][3]

		if minVsMse > (vsMse / 5.0):
			minVsMse = (vsMse / 5.0)
			minVsMseLr = lr

		print lr + ", " + str(tsAcc / 5.0) + ", " + str(tsMse / 5.0) + ", " + str(vsAcc / 5.0) + ", " + str(vsMse / 5.0)

	print "Min vs mse: " + str(minVsMse)
	print "Lr: " + str(minVsMseLr)

def main():
	calcData()
	#plotAccAndMse("part_three.csv")


if __name__ == "__main__":
	main()

