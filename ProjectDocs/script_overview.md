# Scripts and Utilities

Several small utilities live in the `python` and `scripts` directories.

```mermaid
graph TD
    subgraph python
        AD[analyze_data.py]
        RC[rowCount.py]
    end
    subgraph scripts
        PDF[pdf_to_markdown.py]
        XLS[xls_to_csv.py]
    end
```

- **python/** – standalone analysis helpers (`analyze_data.py`, `rowCount.py`).
- **scripts/** – containerized utilities for converting PDFs and spreadsheets.
- `scripts/docker-compose.yml` runs a lightweight service named `pdf_to_md`.

