import java.io.BufferedWriter;
import java.io.FileWriter;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;


public class WeatherForecastImporter {
	
	public static void main(String[] args) throws Exception {	
		Locale.setDefault(Locale.ENGLISH);
		if (args.length < 3) {
			System.err.println("You need to supply starting and ending timestamps and a filename to save to.");
			System.exit(-1);
		}
		
		importScript(args[0], args[1], args[2]);
	}
	
	private static Statement db;
	
	private static void openConnection() throws Exception {
		Connection con = DriverManager.getConnection("jdbc:MySQL://130.225.157.104:3306/weather_forecast_conwx","weather","br4uCect"); 

		//Hvis forbindelsen er der
		if(!con.isClosed())
			//Opret statement objekt til kommunkation med databasen. Dette objekt gives til Reader og Writer
			db = con.createStatement(); 	    
	}
	
	public static ResultSet executeSQL(String sql) throws Exception {
		if (db == null) {
			openConnection();
		}
		return db.executeQuery(sql);
	}
	
	public static void importScript(String start, String end, String filename) throws Exception {
		Map<String, List<String>> data = new TreeMap<String, List<String>>();
		List<String> tmp = new ArrayList<String>();
		tmp.add("timestamp");
		tmp.add("temperature");
		tmp.add("sun_radiation");
		data.put("0", tmp);
		
		
		//temperature
		String sql = "SELECT timest, `value` FROM aarslev_uc55_d0 WHERE timest >= "+ start +" AND timest <= "+ end;
		ResultSet rs = executeSQL(sql);

		//Get data		
		while (rs.next()) {
			long time = rs.getLong("timest");
			String val = rs.getString("value");
				
			List<String> existingData = data.get(Long.toString(time));
			if (existingData == null) {
				existingData = new ArrayList<String>();
				existingData.add(Long.toString(time));
			}
			if (val.equals("-999")) {
				existingData.add(null);
			} else {
				existingData.add(String.format("%.3f", Double.parseDouble(val)));
			}
			data.put(Long.toString(time), existingData);
		}
		
		//sun radiation
		sql = "SELECT timest, `value` FROM aarslev_uc56_d0 WHERE timest >= "+ start +" AND timest <= "+ end;
		rs = executeSQL(sql);
	
		//Get data		
		while (rs.next()) {
			long time = rs.getLong("timest");
			String val = rs.getString("value");
				
			List<String> existingData = data.get(Long.toString(time));
			if (existingData == null) {
				existingData = new ArrayList<String>();
				existingData.add(Long.toString(time));
			}
			if (Double.parseDouble(val) < 0) {
				existingData.add(null);
			} else {
				existingData.add(String.format("%.3f", Double.parseDouble(val)));
			}
			data.put(Long.toString(time), existingData);
		}
		
		//Write data out
		List<List<String>> tmpList = new ArrayList<List<String>>(data.values());
		List<String[]> tmptmpList = new ArrayList<String[]>();
		for (List<String> ss : tmpList) {
			tmptmpList.add(ss.toArray(new String[0]));
		}
				
		//Write data to file to parse it later
		BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
		for (String[] ss : tmptmpList) {
			bw.write(combine(";", ss));
			bw.newLine();
		} 
		bw.flush();
		bw.close();
	}
	
	private static String combine(String glue, String... s)
	{
	  int k = s.length;
	  if ( k == 0 )
	  {
	    return null;
	  }
	  StringBuilder out = new StringBuilder();
	  out.append( s[0] );
	  for ( int x=1; x < k; ++x )
	  {
	    out.append(glue).append(s[x]);
	  }
	  return out.toString();
	}
}
