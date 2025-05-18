# Chroma DB Query Tool

A simple and efficient tool to query your local [Chroma DB](https://www.trychroma.com/) and its collections.  
The data in your Chroma DB is populated by a separate browser scraper repository.

---

## Features

- Query local Chroma DB collections with ease
- Explore and filter data stored in your Chroma instance
- Designed to work seamlessly with data ingested from an external browser scraper

---

## Prerequisites

- [Chroma DB](https://www.trychroma.com/) running locally
- Data ingested into Chroma DB via the companion browser scraper repository
- Python 3.8+ (or specify your language/environment if different)
- (Optional) Virtual environment for Python projects

---

## Setup

1. **Clone this repository:**

   ```bash
   git clone https://github.com/yourusername/chroma-db-query-tool.git
   cd chroma-db-query-tool
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your Chroma DB is running locally and populated with data from the browser scraper.**

---

## Usage

1. **Start the query tool:**

   ```bash
   python main.py
   ```

   _(Replace `main.py` with your entry point if different)_

2. **Query collections:**

   - List available collections
   - Search or filter data by your desired criteria

3. **Example query:**

   ```python
   # Example Python code to query a collection
   from chromadb import Client

   client = Client()
   collection = client.get_collection("your_collection_name")
   results = collection.query("your search term")
   print(results)
   ```

---

## Data Source

- Data is provided by a separate browser scraper repository.
- Make sure to run the scraper and ingest data into your local Chroma DB before querying.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## License

[MIT License](LICENSE)  
_(Or specify your license)_

---

## Acknowledgements

- [Chroma DB](https://www.trychroma.com/)
- The browser scraper project for data ingestion
