import numpy as np
import matplotlib.pyplot as plt

def plotAccAndMse():
	fig, accPlot = plt.subplots()
	t = np.arange(0.01, 10.0, 0.1)
	acc = np.exp(t)
	accPlot.plot(t, acc, "b-")
	accPlot.set_xlabel("epochs")
	accPlot.set_ylabel("accuracy", color="b")
	for tick in accPlot.get_yticklabels():
		tick.set_color("b")

	msePlot = accPlot.twinx()
	mse = np.sin(2 * np.pi * t)
	msePlot.plot(t, mse, "r.")
	msePlot.set_ylabel("mean squared error", color="r")
	for tick in msePlot.get_yticklabels():
		tick.set_color("r")
	plt.show()
	

def main():
	plotAccAndMse()

if __name__ == "__main__":
	main()

