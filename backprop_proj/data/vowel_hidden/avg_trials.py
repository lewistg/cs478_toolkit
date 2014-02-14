import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import BlendedGenericTransform

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
	epochs = [row[5] for row in data]
	epochs = epochs[1:]

	assert(len(tsAcc) == len(tsMse))
	fig, tsAccPlot = plt.subplots()
	fig.subplots_adjust(right=0.75)
	tsAccPlot.plot(learningRate, tsAcc, "b-", linewidth = 2, label = "Accuracy for TS")
	tsAccPlot.plot(learningRate, vsAcc, "g-", linewidth = 2, label = "Accuracy for VS")
	tsAccPlot.set_xlabel("Number of Hidden Nodes", fontsize=18)
	tsAccPlot.set_xscale("log", basex=2)
	tsAccPlot.set_ylabel("Accuracy", fontsize=18, color="b")
	for tick in tsAccPlot.get_yticklabels():
		tick.set_color("b")
	yin, yax = plt.ylim()
	plt.ylim(ymax = yax + .2)
	plt.title("Accuracy, MSE, and Training Time vs. Learning Rate", fontsize=18)

	tsMsePlot = tsAccPlot.twinx()
	tsMsePlot.plot(learningRate, tsMse, "r", linestyle="--", linewidth = 2, label = "MSE for TS")
	tsMsePlot.plot(learningRate, vsMse, "m", linestyle="--", linewidth = 2, label = "MSE for VS")
	yin, yax = plt.ylim()
	plt.ylim(ymax = yax + .1)
	tsMsePlot.set_ylabel("Mean Squared Error (MSE)", color="r", fontsize=18)
	for tick in tsMsePlot.get_yticklabels():
		tick.set_color("r")

	epochsPlot = tsAccPlot.twinx()
	epochsPlot.spines['right'].set_position(('axes', 1.2))
	l5 = epochsPlot.plot(learningRate, epochs, linestyle="-.", color="#FF6600", linewidth = 2, label = "Epochs to Train")
	epochsPlot.set_ylabel("Epochs to Train", color="#FF6600", fontsize=18)
	for tick in epochsPlot.get_yticklabels():
		tick.set_color("#FF6600")
	epochsPlot.set_frame_on(True)
	epochsPlot.patch.set_visible(False)
	#plt.legend(loc=3)
	print epochs

	fig.legend((epochsPlot.lines[0], tsAccPlot.lines[0], tsAccPlot.lines[1], tsMsePlot.lines[0], tsMsePlot.lines[1]), 
	(epochsPlot.lines[0].get_label(), tsAccPlot.lines[0].get_label(), tsAccPlot.lines[1].get_label(), tsMsePlot.lines[0].get_label(), tsMsePlot.lines[1].get_label()), 
	bbox_to_anchor=(-0.25, -0.1, 1, 1))

	plt.show()

def calcData():
	print "LR, TS ACC, TS MSE, VS ACC, VS MSE, EPOCHS"
	minVsMse = 1000
	minVsMseLr = 0
	for powOfTwo in range(1,14): 
		tsAcc = 0.0
		tsMse = 0.0
		vsAcc = 0.0
		vsMse = 0.0
		epochs = 0.0
		for i in range(5):
			data = parseCsvData(str(2**powOfTwo) + "_trial%d.txt" % i)

			# find the record for 100 before the end
			data = np.array(data[1:], dtype=float)
			bestRecord = len(data) - 102
			tsAcc += data[bestRecord][0]
			tsMse += data[bestRecord][1]
			vsAcc += data[bestRecord][2]
			vsMse += data[bestRecord][3]
			epochs += len(data)

		#if minVsMse > (vsMse / 5.0):
			#minVsMse = (vsMse / 5.0)
			#minVsMseLr = lr

		print str(2**powOfTwo) + ", " + str(tsAcc / 5.0) + ", " + str(tsMse / 5.0) + ", " + str(vsAcc / 5.0) + ", " + str(vsMse / 5.0) + ", " + str(epochs / 5.0)

	#print "Min vs mse: " + str(minVsMse)
	#print "Lr: " + str(minVsMseLr)

def main():
	#calcData()
	plotAccAndMse("hidden.csv")


if __name__ == "__main__":
	main()

