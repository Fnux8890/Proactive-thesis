import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class DataCombiner {

	public static void main(String[] args) throws IOException {	
		if (args.length < 3) {
			System.err.println("You need to supply 2 input files and 1 output file, the first file should contain the timestamps in the first column, that should be matched");
			System.exit(-1);
		}
		
		//Check the files
		FileReader targetFile = null;
		FileReader otherFile = null;
		try {
			targetFile = new FileReader(args[0]);
		} catch (FileNotFoundException e) {
			System.err.println("Could not find the first file!");
			System.exit(-1);
		} 
		try {
			otherFile = new FileReader(args[1]);
		} catch (FileNotFoundException e) {
			System.err.println("Could not find the second file!");
			System.exit(-1);
		}
		
		//Create a map for the second file and open the file
		Map<String, String[]> secondFile = new HashMap<String, String[]>();
		CSVReader reader = new CSVReader(otherFile, ';');
		//Skip header, but save it
		String[] header = reader.readNext();
		String[] line;
		while ((line = reader.readNext()) != null) {
			//Remove timestamp
			String timestamp = line[0];
			String[] data = new String[line.length-1];
			System.arraycopy(line, 1, data, 0, data.length);
			//Put it into the map
			secondFile.put(timestamp, data);
		}
		//Close the file
		reader.close();
		otherFile.close();
		
		//Create a target map and open the target file
		List<String[]> target = new ArrayList<String[]>();
		reader = new CSVReader(targetFile, ';');
		//Get header
		line = reader.readNext();
		//Combine headers and put in the list
		String[] totalLine = new String[line.length+header.length-1]; //-1 because the first header still contains timestamp
		System.arraycopy(line, 0, totalLine, 0, line.length);
		System.arraycopy(header, 1, totalLine, line.length, header.length-1);
		target.add(totalLine);
		List<String> missing = new ArrayList<String>();
		while ((line = reader.readNext()) != null) {
			//Get timestamp
			String timestamp = line[0];
			//Copy over to totalLine
			totalLine = new String[totalLine.length];
			System.arraycopy(line, 0, totalLine, 0, line.length);
			//Read from map
			String[] tmp = secondFile.get(timestamp);
			if (tmp == null) {
				//Security check
				missing.add(timestamp);
				continue;
			}
			System.arraycopy(tmp, 0, totalLine, line.length, tmp.length);
			//Save
			target.add(totalLine);
		}
		//Close the file
		reader.close();
		targetFile.close();
		
		//Write out missing timestamps
		if (missing.size() > 0) {
			System.err.println("Timestamps ("+ combine(missing.toArray(new String[0]), ',') +") missing in " + args[1]);
			System.exit(-1);
		}
		
		//Open output file
		BufferedWriter writer = new BufferedWriter(new FileWriter(args[2]));
		//Go through the list and write out, line by line
		boolean first = true;
		for (String[] s : target) {
			if (!first) {
				writer.append(System.lineSeparator());
			}
			first = false;
			writer.append(combine(s,';'));
		}
		//Flush and close
		writer.flush();
		writer.close();
		//Write out that we are done
		System.out.println("Combining complete");
	}

	private static String combine(String[] s, char glue) {
		int k = s.length;
		if ( k == 0 ) {
			return null;
		}
		StringBuilder out = new StringBuilder();
		out.append( s[0] );
		for (int x = 1; x < k; ++x) {
			out.append(glue).append(s[x]);
		}
		return out.toString();
	}

}
