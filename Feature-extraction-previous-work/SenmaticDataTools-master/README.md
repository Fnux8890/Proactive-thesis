Fra Semantic Export tool til sMAP
- Eksporter det korrekte ark fra Excel til .csv fil
- Kør MultipleColumnToOneCsv (java -jar tools/MultipleColumnToOneCsv.jar <input> <output>) - Brug på alle filer
- Kør SenmaticTosMAPCleaner (java -jar tools/SenmaticTosMAPCleaner.jar <input> <output>)
- ?? Eventuelt manglende step
- Kør WeatherForecastImporter (java -jar tools/WeatherForecastImporter.jar <start> <end> <outputfil>) - Start og end fås fra JSONToCsvConverter
- Kør evt Downsampler (java -jar tools/Downsampler.jar <inputfil> <starttime> <endtime> <interval> <outputfil>). Start og end fås fra weather filen. Interval er typisk 3600000 for timeopløsning.
- Kør DataCombiner (java -jar tools/DataCombiner.jar <targetfil> <andenfil> <outputfil>). Typisk vil <targetfil> være vejrfilen, fordi den har mindst opløsning og <andenfil> vil være den fra sMAP. Scriptet oplyser om manglende timestamps.
