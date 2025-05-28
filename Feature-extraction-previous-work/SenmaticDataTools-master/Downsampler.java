import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * @author Morten
 *
 */
public class Downsampler {
	
	//Needs file, start, end, interval and output
	public static void main(String[] args) {
		if (args.length < 5) {
			System.err.println("You need to prove input file, start timestamp, end timestamp, interval in ms and output file");
			System.exit(-1);
		}
		
		long start = Long.parseLong(args[1]);
		long end = Long.parseLong(args[2]);
		long interval = Long.parseLong(args[3]);
		//Build target array
		Map<Long, Reading> readingsOne = new TreeMap<Long, Reading>();
		Map<Long, Reading> readingsTwo = new TreeMap<Long, Reading>();
		for (long l = start; l <= end; l += interval) {
			readingsOne.put(l, new Reading(0,0));
			readingsTwo.put(l, new Reading(0,0));
		}
		//Read input file
		CSVReader reader = null;
		try {
			reader = new CSVReader(new FileReader(args[0]), ';');
		} catch (FileNotFoundException e) {
			System.err.println("Could not find input file!");
			System.exit(-1);
		}
		String[] line = null;
		try {
			line = reader.readNext();
		} catch (IOException e) {
			System.err.println("Could not read from input file!");
			System.exit(-1);
		}
		//Save header
		String[] header = line;
		try {
			while ((line = reader.readNext()) != null) {
				long time = Long.parseLong(line[0]);
				//Convert to a calendar
				Calendar c = Calendar.getInstance(Locale.forLanguageTag("da-DK"));
				c.setTimeInMillis(time);
				//Copy twice
				Calendar below = (Calendar) c.clone();
				below.set(Calendar.MINUTE, 0);
				below.set(Calendar.SECOND, 0);
				below.set(Calendar.MILLISECOND, 0);
				Calendar above = (Calendar) c.clone();
				above.set(Calendar.HOUR_OF_DAY, c.get(Calendar.HOUR_OF_DAY)+1);
				above.set(Calendar.MINUTE, 0);
				above.set(Calendar.SECOND, 0);
				above.set(Calendar.MILLISECOND, 0);
				//Find closest hour's timestamp
				long downDiff = c.getTimeInMillis() - below.getTimeInMillis();
				long upDiff = above.getTimeInMillis() - c.getTimeInMillis();
				long targetTime = (downDiff < upDiff ? below : above).getTimeInMillis();
				//Get the reading from target array
				Reading r = readingsOne.get(targetTime);
				//Safety check
				if (r == null) { continue; }
				boolean overwrite = false;
				//check the reading
				if (r.getValue() == 0) {
					overwrite = true;
					//No reading exists, so we overwrite
				}
				if (Math.abs(targetTime-r.getTimeStamp()) > Math.abs(targetTime-time)) {
					//The current reading is closer, so overwrite
					overwrite = true;
				}
				if (overwrite) {
					readingsOne.put(targetTime, new Reading(time, Double.parseDouble(line[1])));
					readingsTwo.put(targetTime, new Reading(time, Double.parseDouble(line[2])));
				}
			}
		} catch (Exception e) {
			System.err.println("Could read or parse input file! " + e.getMessage());
			System.exit(-1);
		}
		
		//Write out again
		Locale.setDefault(Locale.ENGLISH);
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(args[4]));
		} catch (IOException e) {
			System.err.println("Could not open output file!");
			System.exit(-1);
		}
		try {
			//Write header
			bw.append(combine(header, ';'));
		} catch (IOException e1) {
			System.err.println("Could not write to output file!");
			System.exit(-1);
		}
		for (long l = start; l <= end; l += 3600000l) {
			try {
				bw.newLine();
				bw.append(String.format("%d;%.3f;%.3f", l, readingsOne.get(l).getValue(), readingsTwo.get(l).getValue()));
			} catch (IOException e) {
				System.err.println("Could not write to output file!");
				System.exit(-1);
			}
		}
		try {
			bw.flush();
			bw.close();
			reader.close();
		} catch (IOException e) {
			System.err.println("Could not close files!");
			System.exit(-1);
		}
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
