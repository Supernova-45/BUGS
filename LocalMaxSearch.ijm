for (prominence = 6; prominence <= 16; prominence = prominence + 2) {
	run("Select None");
	run("Clear Results");
	for (slice = 1; slice <= 42; slice++){
		setSlice(slice);
		run("Find Maxima...", "prominence="+prominence+" output=[Point Selection]");
		run("Measure");
		run("Select None");
	}
	saveAs("Results", "/Users/alexandrakim/Desktop/BUGS2022/BUGScode/data/local_max/local_max_2P_prominence_"+prominence+".csv");
}