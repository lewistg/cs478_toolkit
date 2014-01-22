import csv

def createLatexTable(tableData):
	"""
	@param tableData - 2D array of data; it is assumed the table headers 
	are in the array
	"""
	tableStr = "\\begin{tabular}{"
	for i in range(len(tableData[0])):
		tableStr += "c"
	tableStr += "}\n"
	tableStr += "\\toprule\n"

	for row in range(len(tableData)):
		if row == 1:
			tableStr += "\\hline\n"
		for col in range(len(tableData[row])):
			tableStr += str(tableData[row][col])
			if col < len(tableData[row]) - 1:
				tableStr += "\t&\t"
			else:
				tableStr += "\\\\\n"
			
	tableStr += "\\bottomrule\n"
	tableStr += "\\end{tabular}"
	return tableStr

def parseCsvData(csvFileName):
	csvData = []
	with open(csvFileName, "rb") as csvfile:
		csvReader = csv.reader(csvfile, delimiter=",")
		for row in csvReader:
			csvData.append(row)
	return csvData

def linSepTables():
	linSepTable = [["x", "y", "Color"], [-.23,.32,"Red"], [-.11,.48,"Red"], [.23,.90,"Red"], [-.14,.83,"Red"], [.36,.61,"Blue"], [.44,.67,"Blue"], [.13,-.12,"Blue"], [.25,-.13,"Blue"]]
	print createLatexTable(linSepTable)

def votingTask():
	votingData = parseCsvData("voting_accuracy.csv")
	print createLatexTable(votingData)

	modelData = parseCsvData("final_models.csv")
	print createLatexTable(modelData)

	avgModelData = parseCsvData("voting_average_model.csv")
	print createLatexTable(avgModelData)

def irisTask():
	votingData = parseCsvData("iris_accuracy.csv")
	print createLatexTable(votingData)

	modelData = parseCsvData("iris_model.csv")
	print createLatexTable(modelData)

	modelData = parseCsvData("avg_iris_model.csv")
	print createLatexTable(modelData)

def myExperiment():
	modelData = parseCsvData("nlt_models.csv")
	print createLatexTable(modelData)

def main():
	#linSepTables()
	#votingTask()
	#irisTask()
	myExperiment()

if __name__ == "__main__":
	main()
