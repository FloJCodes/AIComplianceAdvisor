# AIComplianceAdvisor

## Still in Progress

Working right now:
- Parsing local .html file of EU AI Act in data folder (BeautifulSoup Scrape was blocked)
- Sortig Act into Annexes, Chapters, Recitals and Chapters into Articles
- Splitting Text into chunks with overlaps for keeping context
- Creating embeddings with --mode local or azure
- local uses sentence-transformers "all-MiniLM-L6-v2" (dimension: 384)
- azure uses model text-embedding-ada-002 (dimension: 1536)