import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;


public class SenmaticTosMAPCleaner {

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
		
		//Open input file
		FileReader fr = null;
		try {
			fr = new FileReader(args[0]);
		} catch (FileNotFoundException e) {
			System.err.println("Could not load input file!");
			System.exit(-1);
		}
		//Create reader
		CSVReader reader = new CSVReader(fr,';');
		//Create output file
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(args[1]));
		} catch (IOException e) {
			System.err.println("Could open load output file!");
			System.exit(-1);
		}
		//Get header
		String[] line = null;
		try {
			line = reader.readNext();
		} catch (IOException e) {
			System.err.println("Could read line from input file!");
			System.exit(-1);
		}
		//Write header
		try {
			bw.append(combine(line,';'));
		} catch (IOException e1) {
			System.err.println("Could write to output file!");
			System.exit(-1);
		}
		//Go through rest of the file
		String s = null;
		try {
			while ((line = reader.readNext()) != null) {
				boolean add = true;
				//Go through each string
				for (int i = 0; i < line.length; i++) {
					s = line[i];
					
					//Put in number if empty
					if (s.trim().equals("")) {
						s = "0.0";
					}
					//Change decimal operator
					s = s.replace(',', '.');
					//Stop if we have any #NUM!
					if (s.equals("#NUM!")) {
						add = false;
						break;
					}
					
					//Stop if we have any crazy numbers
					if (i > 0 && Double.parseDouble(s) > 1.0E+30) {
						add = false;
						break;
					}
					
					//Replace in array
					line[i] = s;
				}
				if (!add) {
					continue;
				}
				//Write to file
				bw.append(System.lineSeparator());
				bw.append(combine(line, ';'));
			}
		} catch (NumberFormatException e) {
			System.err.println("Could not parse number in input file: " + e.getMessage());
			System.exit(-1);
		} catch (IOException e) {
			System.err.println("Could write to output file!");
			System.exit(-1);
		}
		//Close everything
		try {
			reader.close();
			bw.flush();
			bw.close();
		} catch (IOException e) {
			System.err.println("Could not close files!");
			System.exit(-1);
		}
		
		//Write all ok
		System.out.println("All ok!");
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
