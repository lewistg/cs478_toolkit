import numpy as np
import matplotlib.pyplot as plt

def partOne():
	xRed = [-.23, -.11, .23, -.14]
	yRed = [.32, .48, .90, .83]
	plt.plot(xRed, yRed, 'ro')

	xBlue = [.36, .44, .13, .25]
	yBlue = [.61, .67, -.12, -.13]
	plt.plot(xBlue, yBlue, 'bo')

	plt.title("Linearly Seprable Points")

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
		yLine.append((17.0/5.0) * x)
	plt.plot(xLine, yLine)

	plt.axis([-.3, .6, -.2, 1])
	plt.show()

def main():
	partOne()
	"""
	plt.plot([3, 2, 4], [5, 8, 10], 'ro')
	plt.plot([7, 8, 8], [5, 8, 10], 'bo')
	plt.axis([0, 10, -10, 20])
	plt.text(4, 10, "4, 10")
	plt.show()
	"""

if __name__ == "__main__":
	main()
