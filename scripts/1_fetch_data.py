# Download from ArXiv
# scripts/1_fetch_data.py
import arxiv
import json
from datetime import datetime
import sys
import os

# Load categories from JSON file
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts/categories.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    CATEGORIES = config['categories']
    START_YEAR = config['start_year']

def fetch_papers_by_category(
    category: str,
    start_year: int,
    max_results_per_category: int = None  # None = fetch all available
):
    """
    Fetch papers from a single ArXiv category
    """
    papers = []
    seen_ids = set()  # Track duplicate paper IDs
    
    # Use Client for newer API with rate limiting
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,  # Be nice to ArXiv API
        num_retries=3
    )
    
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results_per_category if max_results_per_category else 20000,  # ArXiv's practical limit
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    print(f"  Fetching {category}...")
    
    try:
        count = 0
        for result in client.results(search):
            count += 1
            
            # Filter by year
            pub_year = result.published.year
            if pub_year < start_year:
                continue
            
            # Skip duplicates
            arxiv_id = result.entry_id.split('/')[-1]
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            
            # Use result.summary instead of result.abstract
            abstract = result.summary if hasattr(result, 'summary') else getattr(result, 'abstract', '')
                
            paper = {
                'arxiv_id': arxiv_id,
                'title': result.title,
                'abstract': abstract,
                'authors': [author.name for author in result.authors],
                'published': result.published.isoformat(),
                'categories': list(result.categories),
                'primary_category': result.primary_category
            }
            papers.append(paper)
            
            if len(papers) % 100 == 0:
                print(f"    Fetched {len(papers)} papers from {category} (processed {count} total)...")
                
    except arxiv.UnexpectedEmptyPageError as e:
        print(f"    ⚠️  Empty page encountered for {category} after {len(papers)} papers")
    except Exception as e:
        print(f"    ⚠️  Error fetching {category}: {e}")
    
    print(f"    ✅ {len(papers)} papers from {category}")
    return papers

def fetch_arxiv_papers(
    categories=None,
    start_year=None,
    fetch_all=True  # If True, fetches all available papers by querying each category separately
):
    """
    Fetch papers from ArXiv in specified categories.
    
    Strategy: Fetch each category separately and combine results.
    This avoids the empty page issues that occur with large OR queries.
    
    Args:
        categories: List of arXiv categories. If None, uses categories from categories.py
        start_year: Year to start fetching from. If None, uses START_YEAR from categories.py
        fetch_all: If True, fetches all available papers by querying each category separately
    """
    # Use defaults from categories.py if not provided
    if categories is None:
        categories = CATEGORIES
    if start_year is None:
        start_year = START_YEAR
    
    all_papers = []
    seen_ids = set()  # Track duplicates across categories
    
    print(f"Fetching papers from ArXiv (categories: {', '.join(categories)})...")
    print(f"Start year: {start_year}")
    print(f"Strategy: Fetching each category separately to avoid API issues\n")
    
    # Fetch each category separately
    for category in categories:
        category_papers = fetch_papers_by_category(category, start_year)
        
        # Add papers, avoiding duplicates
        for paper in category_papers:
            if paper['arxiv_id'] not in seen_ids:
                seen_ids.add(paper['arxiv_id'])
                all_papers.append(paper)
        
        print(f"  Total unique papers so far: {len(all_papers)}\n")
    
    print(f"\n{'='*60}")
    print(f"✅ Total unique papers fetched: {len(all_papers)}")
    print(f"{'='*60}")
    
    return all_papers

if __name__ == "__main__":
    # Fetch all papers by querying each category separately
    # This avoids ArXiv API issues with large OR queries
    # Categories and start_year are loaded from categories.py by default
    papers = fetch_arxiv_papers(
        fetch_all=True  # Fetch all available papers
    )
    
    # Save to JSON
    if papers:
        with open('data/papers.json', 'w') as f:
            json.dump(papers, f, indent=2)
        print(f"\n✅ Saved {len(papers)} unique papers to data/papers.json")
    else:
        print("\n❌ No papers were fetched. Please try again later or check your query.")