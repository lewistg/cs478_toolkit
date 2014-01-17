
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


def main():
	table = [["dog", "cat"], [5, 3]]
	print createLatexTable(table)

if __name__ == "__main__":
	main()
