# Use NimbleCSV stream parsing directly with specified options
file_content
|> NimbleCSV.parse_string(separator: ";", escape: "\\", headers: true)
|> Stream.map(&convert_row_to_map(&1, headers))
|> Stream.chunk_every(@batch_size)
