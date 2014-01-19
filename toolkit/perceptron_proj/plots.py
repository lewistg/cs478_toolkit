import numpy as np
import matplotlib.pyplot as plt

def partOne():
	xRed = [-.23, -.11, .23, -.14]
	yRed = [.32, .48, .90, .83]
	plt.plot(xRed, yRed, 'ro')

	xBlue = [.36, .44, .13, .25]
	yBlue = [.61, .67, -.12, -.13]
	plt.plot(xBlue, yBlue, 'bo')

	plt.title("Linearly Seprable Points", fontsize=24)

	for i in range(len(xRed)):
		plt.text(xRed[i] + -0.06, yRed[i] - 0.05, "(" + str(xRed[i]) + ", " + str(yRed[i]) + ")")
		if i == 2:
			plt.text(xBlue[i] + -0.1, yBlue[i] - 0.05, "(" + str(xBlue[i]) + ", " + str(yBlue[i]) + ")")
		else:
			plt.text(xBlue[i] + -0.06, yBlue[i] - 0.05, "(" + str(xBlue[i]) + ", " + str(yBlue[i]) + ")")
	
	xLine = []
	yLine = []
	for i in range(-3, 11):
		x = i / 10.0
		xLine.append(x)
		#yLine.append((17.0/5.0) * x)
		yLine.append((162.0/7.0) * x)
	plt.plot(xLine, yLine)

	plt.xlabel("x", fontsize=18)
	plt.ylabel("y", fontsize=18)
	plt.axis([-.3, .6, -.2, 1])
	plt.show()

def nonLinSepPoints():
	xRed = [.15, -.14, -.18, .15]
	yRed = [-.32, .13 , -.15, .17]
	plt.plot(xRed, yRed, 'ro')

	xBlue = [.15, -.16, .15, -.18]
	yBlue = [-.50, -.35, -.11, -.25]
	plt.plot(xBlue, yBlue, 'bo')

	for i in range(len(xRed)):
		plt.text(xRed[i] + -0.06, yRed[i] - 0.05, "(" + str(xRed[i]) + ", " + str(yRed[i]) + ")")
		if i == 2:
			plt.text(xBlue[i] + -0.1, yBlue[i] - 0.05, "(" + str(xBlue[i]) + ", " + str(yBlue[i]) + ")")
		else:
			plt.text(xBlue[i] + -0.06, yBlue[i] - 0.05, "(" + str(xBlue[i]) + ", " + str(yBlue[i]) + ")")

	xLine = []
	yLine = []
	for i in range(-7, 11):
		x = i / 10.0
		xLine.append(x)
		yLine.append((33.0/161.0) * x)
	plt.plot(xLine, yLine)

	plt.title("Non-linearly Seprable Points", fontsize=24)
	plt.xlabel("x", fontsize=18)
	plt.ylabel("y", fontsize=18)
	plt.axis([-.5, .5, -.6, .5])
	plt.show()

def partLearningRate():
	learningRate = [.02, .04, .06, .08, .1, .12, .14, .16, .18, .2]

	# linearly seprable	
	linSepEpochs = [9.6, 8.6, 9, 8.4, 10, 8.8, 10.6, 9.6, 9.4, 9]
	plt.plot(learningRate, linSepEpochs, "o-", label = "Linearly seprable set")

	# non-linearly seprable
	nonLinSepEpochs = [46.6, 39, 47.4, 41, 29.2, 25.2, 33.8, 37.4, 37.6, 39.2]
	plt.plot(learningRate, nonLinSepEpochs, "o-", label = "Non-linearly seprable set")

	plt.legend()

	plt.title("Effect of Learning Rate on Epochs", fontsize=24)
	plt.xlabel("Learning Rate", fontsize=18)
	plt.ylabel("Number of Epochs", fontsize=18)
	plt.show()

def nonStopIris():

def main():
	partOne()
	#nonLinSepPoints()
	#partLearningRate()

if __name__ == "__main__":
	main()
