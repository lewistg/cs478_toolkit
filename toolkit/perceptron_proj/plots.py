import numpy as np
import random
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

	plotLineProgression("lin_sep_training_progression")

	plt.xlabel("x", fontsize=18)
	plt.ylabel("y", fontsize=18)
	plt.axis([-.3, .6, -.2, 1])
	plt.show()

def plotLineProgression(fileName):
	# create all of the lines
	f = open(fileName, "r")
	lines = []
	for line in f:
		numStrs = line.split(",")
		lines.append(numStrs)
		
	for i in range(len(lines)):
		#numStrs = line.split(",")
		line = lines[i]
		if float(line[1]) == 0:
			continue
		m = float(line[0]) / -float(line[1])
		b = float (line[2]) / -float(line[1])
		xLine = []
		yLine = []
		for j in range(-7, 11):
			x = j / 10.0
			xLine.append(x)
			yLine.append(m * x + b)
		greyScale = 1.0 - float(i) / (len(lines))
		print greyScale
		print i
		if i < len(lines) - 1:
			plt.plot(xLine, yLine, color=str(greyScale))
		else:
			plt.plot(xLine, yLine, linewidth=4.0)

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
	
	# create all of the lines
	plotLineProgression("line_progression_non_lin")

	plt.title("Nonlinearly Seprable Points", fontsize=24)
	plt.xlabel("x", fontsize=18)
	plt.ylabel("y", fontsize=18)
	plt.axis([-.5, .5, -.6, .5])
	plt.show()

def partLearningRate():
	learningRate = [.02, .04, .06, .08, .1, .12, .14, .16, .18, .2]

	# linearly seprable	
	linSepEpochs = [9.8,9,9,9.4,9.4,10,9,7.8,7.8,10.4]

	plt.plot(learningRate, linSepEpochs, "o-", label = "Linearly seprable set")

	# non-linearly seprable
	nonLinSepEpochs = [26.6,37,17.8,38,49.4,31.8,43.6,55.6,42.6,31]

	plt.plot(learningRate, nonLinSepEpochs, "o-", label = "Non-linearly seprable set")

	plt.legend(loc=0)

	plt.title("Effect of Learning Rate on Epochs", fontsize=24)
	plt.xlabel("Learning Rate", fontsize=18)
	plt.ylabel("Epochs to Train", fontsize=18)
	plt.axis([.02, .2, 0, 60])
	plt.show()

def versusErrorRate():
	errorRate = [0.07722338,0.04815616,0.05336226,0.07939262,0.06334054,0.04078092,0.04772232,0.0451193,0.03991326,0.05726674,0.03600872,0.0494577,0.0347072,0.05379612,0.07114962,0.04295012,0.03774406,0.059436,0.04598696,0.03340568,0.04034708,0.0368764,0.04164862,0.05813458,0.03861174,0.03687638,0.0563992,0.04425164,0.04728852,0.0468547,0.0581345,0.05943596,0.04338396,0.03557488,0.0659436,0.05292846,0.04381782,0.05596528,0.03817788,0.04078094,0.05639908,0.04555314,0.06160518,0.03904556,0.04642084,0.04945772,0.03687638,0.0555315,0.04078092,0.04598698,0.05466376,0.04338396,0.05509762,0.04208244,0.04251628,0.04772236,0.08026028,0.05422994,0.04989154,0.0381779,0.03557488,0.04815622,0.05249458,0.04511932,0.04425166,0.0416486,0.0381779,0.04642082,0.03427334,0.04034708,0.03774404,0.04164862,0.06334048,0.06203898,0.06160532,0.03991324,0.03557486,0.0394794,0.03557488,0.0481562,0.03427336,0.03731024,0.04728844,0.03687636,0.04034708,0.032538,0.04338396,0.03427334,0.04728852,0.06117128,0.03644256,0.04208244,0.04642084,0.0416486,0.03904558,0.04251626,0.03904558,0.0416486,0.05639914,0.0390456]


	plt.title("Error Rate vs. Training Time", fontsize=24)
	plt.xlabel("Epoch", fontsize=18)
	plt.ylabel("Error Rate", fontsize=18)
	#plt.plot([i for i in range(1, 87)], errorRate)
	plt.axis([0, 100, 0, 1])
	plt.plot(errorRate)
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

	plt.title("Fitting Preprocessed Input", fontsize=24)
	plt.xlabel("x", fontsize=18)
	plt.ylabel("y", fontsize=18)

	plt.show()

def nltPlot2(): 
	yellowX = [random.random() for x in range(4)] 
	yellowY = [random.random() for y in range(4)]


	greenX = [random.random() for x in range(4)] 
	greenY = [random.random() for y in range(4)]


	for i in range(4):
		redX = [random.gauss(yellowX[i], 0.1) for j in range(100)]
		redY = [random.gauss(yellowY[i], 0.1) for j in range(100)]
		plt.plot(redX, redY, "ro")

	for i in range(4):
		redX = [random.gauss(greenX[i], 0.1) for j in range(100)]
		redY = [random.gauss(greenY[i], 0.1) for j in range(100)]
		plt.plot(redX, redY, "bo")

	plt.plot(yellowX, yellowY, "yo", markersize = 15)
	plt.plot(greenX, greenY, "go", markersize = 15)

	makeArffFiles(redX, redY, "red")
	makeArffFiles(blueX, blueY, "blue")

	plt.show()

def makeArffFiles(x, y, label):
	for n in range(1, 101):
		print n
		f = open("experiment/nlt2%d.arff" % n, "w")
		f.write("@relation nonlinsep\n")
		for i in range((n +1)**2 - 1):
			f.write("@attribute xy%d real\n" % i)
		f.write("@attribute color {red, blue}\n")
		f.write("@data\n")
		for pti in range(len(x)):
			f.write(transform(x[pti], y[pti], n) + "," + label + "\n")

def transform(x, y, n):
	features = ""
	for i in range(n + 1):
		for j in range(n + 1):
			if(i != 0 or j != 0):
				features += str(x**i * y**j)
				if (i < n) or (j < n):
					features += ","
	return features

def main():
	#partOne()
	#nonLinSepPoints()
	#partLearningRate()
	versusErrorRate()
	#nltPlot()
	#nltPlot2()

if __name__ == "__main__":
	main()
