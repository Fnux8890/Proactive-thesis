import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class MultipleColumnToOneCsv {

	public static void main(String[] args) {
		//Check if we have * in first arg, which means that first arg is a folder an it should take every file in that folder with correct ending
		if (args.length == 1 && args[0].endsWith("*.csv")) {
			//Cut away ending
			String folderPath = args[0].substring(0, args[0].length()-5);
			//Make sure it is a directory
			File dir = new File(folderPath);
			if (!dir.isDirectory()) {
				System.err.println("Folder name not given. Ending!");
				System.exit(-1);
			}
			String[] files = dir.list(new FilenameFilter() {
				public boolean accept(File dir, String name) {
					if (name.endsWith(".csv"))
						return true;
					return false;
				}
			});
			System.out.println("Starting directory conversion");
			//Call the main method again but with new arguments
			for (String file : files) {
				//Get output file name
				file = dir.getAbsolutePath() + "/" + file;
				String output = file.substring(0, file.length()-4) + "_.csv";
				main(new String[] {file,output});
			}
			System.out.println("Directory conversion done!");
			System.exit(-1);
		}
		
		
		if (args.length < 2) {
			System.err.println("You need to supply 1 input file and 1 output file"); 
			System.exit(-1);
		}
		
		//Load input file
		FileReader fr = null;
		try {
			fr = new FileReader(args[0]);
		} catch (FileNotFoundException e) {
			System.err.println("Could not load input file!");
			System.exit(-1);
		}
		CSVReader reader = new CSVReader(fr, ';');
		//Figure out if there are any empty fields (indicates multiple columns)
		String[] line = null;
		try {
			line = reader.readNext();
		} catch (IOException e) {
			System.err.println("Could not next line in input file");
			System.exit(-1);
		}
		int size = line.length;
		for (int i = 0; i < line.length; i++) {
			if (line[i].trim().equals("") && i > 0) { //first is empty because its timestamp
				size = i+1; //+1 because there is 1 empty behind
				break;
			}
		}
		//Split
		List<String[]> header = chunk(line, size);
		//we only need first ones of the header, because they are the same
		line = header.get(0);
		//Create writer
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(args[1]));
		} catch (IOException e) {
			System.err.println("Could not open output file!");
			System.exit(-1);
		}
		//Write hedaer
		try {
			bw.append(combine(line, ';'));
		} catch (IOException e) {
			System.err.println("Could not write to output file!");
			System.exit(-1);
		}
		//Go through rest of file
		List<String[]> chunks;
		try {
			while ((line = reader.readNext()) != null) {
				//Split into chucnks
				chunks = chunk(line, size);
				//Go through each chunk
				for (String[] ch : chunks) {
					//Check if empty
					if (ch[0].trim().equals("")) {
						continue;
					}
					//Write to file (this is ok, because order doesnt matter)
					bw.append(System.lineSeparator());
					bw.append(combine(ch,';'));
				}
			}
		} catch (IOException e) {
			System.err.println("Could write to output file!");
			System.exit(-1);
		}
		//Close
		try {
			bw.flush();
			bw.close();
			reader.close();
		} catch (IOException e) {
			System.err.println("Could not close files!");
			System.exit(-1);
		}
		
		//Write ok
		System.out.println("All ok!");
	}

	
	private static <T> List<T[]> chunk(T[] arr, int size) {
	    if (size <= 0)
	        throw new IllegalArgumentException("Size must be > 0 : " + size);

	    List<T[]> result = new ArrayList<T[]>();

	    int from = 0;
	    int to = size >= arr.length ? arr.length : size;

	    while (from < arr.length) {
	        T[] subArray = Arrays.copyOfRange(arr, from, to);
	        from = to;
	        to += size;
	        if (to > arr.length) {
	            to = arr.length;
	        }
	        result.add(subArray);
	    }
	    //On all but the last one, remove last
	    List<T[]> finalResult = new ArrayList<T[]>();
	    for (int i = 0; i < result.size(); i++) {
	    	T[] t = result.get(i);
	    	if (i == result.size()-1) {
	    		finalResult.add(t);
	    		break;
	    	}
	    	finalResult.add(Arrays.copyOfRange(arr, 0, t.length-1));
	    }
	    return finalResult;
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
