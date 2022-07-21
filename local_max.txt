for (prominence = 6; prominence <= 16; prominence = prominence + 2) {
for (slice = 1; slice <= 42; slice++){
setSlice(slice);
run("Find Maxima...", "prominence="+prominence+" output=[Point Selection]");
run("Measure");
run("Select None");}
saveAs("Results", "/Users/alexandrakim/Desktop/BUGS2022/local_max_2P_prominence_"+prominence+".csv");
}