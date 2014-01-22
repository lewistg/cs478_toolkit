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

def versusErrorRate():
	errorRate = [0.0773994,0.0390093,0.09535602,0.04272448,0.0817338,0.047678,0.04582046,0.04210528,0.03405572,0.04272442,0.0735294,0.0410217,0.030959775,0.035603725,0.034055725,0.040247675,0.1106811,0.039473675,0.0549536,0.0361197333,0.0515996333,0.0350877333,0.0371517,0.0319917667,0.0319917667,0.0794634667,0.0330237667,0.0309597667,0.0371517333,0.0526316,0.0350877333,0.0330237333,0.0330237667,0.0650154667,0.0402476667,0.0619195,0.0340557667,0.03560375,0.0263158,0.0495356,0.04489165,0.0356037,0.0619195,0.04643965,0.0371517,0.03560375,0.0402477,0.0278638,0.0278638,0.0464396,0.0278638,0.0743034,0.0309598,0.0247678,0.0247678,0.0278638,0.0278638,0.0216718,0.0340557,0.0495356,0.0309598,0.0216718,0.0216718,0.0247678,0.111455,0.0464396,0.142415,0.0309598,0.0588235,0.0309598,0.0340557,0.0371517,0.0340557,0.0433437,0.0247678,0.0309598,0.0278638,0.0247678,0.0402477,0.0247678,0.0309598,0.0247678,0.0216718,0.0247678,0.0185759,0.0216718]
	plt.title("Error Rate vs. Epochs", fontsize=24)
	plt.xlabel("Epoch", fontsize=18)
	plt.ylabel("Error Rate", fontsize=18)
	plt.plot([i for i in range(1, 87)], errorRate)
	plt.show()

def nltPlot(): 
	redPts = [(-2.34, .9677), (2.128, .3226), (2.766, 3.226), (1.7021, 6.129), (-1.914894, -1.290323), (1.2766, 2.258), (1.9148936, .9677)]
	bluePts = [(-.21277, 3.548), (2.1277, -1.9355), (.8511, -3.2258), (-.6384, -4.8387), (1.7021, -3.5484), (4.2553, 2.581), (.42553191, 1.9355)]

	plt.plot([pt[0] for pt in redPts], [pt[1] for pt in redPts], "ro")
	plt.plot([pt[0] for pt in bluePts], [pt[1] for pt in bluePts], "bo")

	xLine = []
	yLine = []
	for i in range(-30, 50):
		x = i / 10.0
		xLine.append(x)
		#yLine.append((17.0/5.0) * x)
		yLine.append(x**3 - 3 * x**2 - x + 5)
	plt.plot(xLine, yLine, label="Original")

	models = [[4.38409,2.90633,-3.84122,9.51148,-10.9],
		[5.34182,3.06776,-3.93253,9.24562,-11],
		[5.17142,3.58455,-4.57759,11.1096,-12.6],
		[5.91608,3.64871,-4.92576,11.6396,-12.3],
		[6.66132,4.23084,-5.01142,11.8543,-15.3]]

	trial = 1
	for model in models:	
		xLine = []
		yLine = []
		for i in range(-30, 50):
			x = i / 10.0
			xLine.append(x)
			y = model[0] * x
			y += model[2] * x**3
			y += model[3] * x**2
			y += model[4]
			y /= -model[1]
			yLine.append(y)
		plt.plot(xLine, yLine, "--", label = "Trial " +  str(trial))
		trial += 1

	plt.legend()

	plt.axis([-3, 5, -5, 7])

	plt.title("Fitting Data after Non-Linear Transform", fontsize=24)
	plt.xlabel("X", fontsize=18)
	plt.ylabel("Y", fontsize=18)

	plt.show()

def main():
	#partOne()
	#nonLinSepPoints()
	#partLearningRate()
	#versusErrorRate()
	nltPlot()

if __name__ == "__main__":
	main()
