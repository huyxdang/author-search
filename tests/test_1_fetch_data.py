"""
Comprehensive unit tests for scripts/1_fetch_data.py
Includes edge cases and error handling.
"""
import unittest
from unittest.mock import Mock, patch, mock_open
import json
import os
import sys
import importlib.util
import arxiv

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Mock config for categories.json
mock_config_json = json.dumps({
    'categories': ['cs.AI', 'cs.LG'],
    'start_year': 2020
})


def load_module(module_name, file_path):
    """Helper to load a module from file"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFetchPapersByCategory(unittest.TestCase):
    """Test fetch_papers_by_category function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_result = Mock()
        self.sample_result.entry_id = 'http://arxiv.org/abs/1234.5678'
        self.sample_result.title = 'Test Paper'
        self.sample_result.summary = 'Test abstract'
        self.sample_result.authors = [Mock(name='John Doe'), Mock(name='Jane Smith')]
        self.sample_result.published = Mock(year=2023)
        self.sample_result.published.isoformat.return_value = '2023-01-15T00:00:00Z'
        self.sample_result.categories = ['cs.AI']
        self.sample_result.primary_category = 'cs.AI'
        self.module_path = os.path.join(project_root, 'scripts', '1_fetch_data.py')
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_fetch_papers_by_category_single_year(self, mock_file):
        """Test fetching papers for a single year"""
        with patch('arxiv.Client') as mock_client_class, \
             patch('arxiv.Search') as mock_search, \
             patch('datetime.datetime') as mock_datetime:
            
            fd = load_module('fetch_data', self.module_path)
            
            mock_datetime.now.return_value.year = 2023
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.results.return_value = iter([self.sample_result])
            mock_search.return_value = Mock()
            
            result = fd.fetch_papers_by_category('cs.AI', 2023)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['arxiv_id'], '1234.5678')
            self.assertEqual(result[0]['title'], 'Test Paper')
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_fetch_papers_by_category_duplicate_handling(self, mock_file):
        """Test that duplicates within a category are removed"""
        with patch('arxiv.Client') as mock_client_class, \
             patch('arxiv.Search') as mock_search, \
             patch('datetime.datetime') as mock_datetime:
            
            fd = load_module('fetch_data', self.module_path)
            
            mock_datetime.now.return_value.year = 2023
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.results.return_value = iter([self.sample_result, self.sample_result])
            mock_search.return_value = Mock()
            
            result = fd.fetch_papers_by_category('cs.AI', 2023)
            
            self.assertEqual(len(result), 1)
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_fetch_papers_by_category_empty_page_error(self, mock_file):
        """Test handling of UnexpectedEmptyPageError"""
        with patch('arxiv.Client') as mock_client_class, \
             patch('arxiv.Search') as mock_search, \
             patch('datetime.datetime') as mock_datetime:
            
            fd = load_module('fetch_data', self.module_path)
            
            mock_datetime.now.return_value.year = 2024
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_search.return_value = Mock()
            
            result_2023 = Mock()
            result_2023.entry_id = 'http://arxiv.org/abs/2023.1234'
            result_2023.title = 'Paper 2023'
            result_2023.summary = 'Abstract'
            result_2023.authors = [Mock(name='Author')]
            result_2023.published = Mock(year=2023)
            result_2023.published.isoformat.return_value = '2023-01-01T00:00:00Z'
            result_2023.categories = ['cs.AI']
            result_2023.primary_category = 'cs.AI'
            
            mock_client.results.side_effect = [
                iter([result_2023]),
                arxiv.UnexpectedEmptyPageError("", 0, None)
            ]
            
            result = fd.fetch_papers_by_category('cs.AI', 2023)
            
            self.assertEqual(len(result), 1)
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_fetch_papers_by_category_no_summary_fallback(self, mock_file):
        """Test handling when result has no summary attribute"""
        with patch('arxiv.Client') as mock_client_class, \
             patch('arxiv.Search') as mock_search, \
             patch('datetime.datetime') as mock_datetime:
            
            fd = load_module('fetch_data', self.module_path)
            
            mock_datetime.now.return_value.year = 2023
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_search.return_value = Mock()
            
            # Create a simple object-like class that doesn't have summary
            class ResultWithoutSummary:
                def __init__(self):
                    self.entry_id = 'http://arxiv.org/abs/1234.5678'
                    self.title = 'Test Paper'
                    self.abstract = 'Test abstract fallback'
                    self.authors = [Mock(name='Author')]
                    self.published = Mock(year=2023)
                    self.published.isoformat = lambda: '2023-01-01T00:00:00Z'
                    self.categories = ['cs.AI']
                    self.primary_category = 'cs.AI'
            
            result_no_summary = ResultWithoutSummary()
            
            # hasattr(result_no_summary, 'summary') will return False since it doesn't exist
            
            mock_client.results.return_value = iter([result_no_summary])
            
            result = fd.fetch_papers_by_category('cs.AI', 2023)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['abstract'], 'Test abstract fallback')


class TestFetchArxivPapers(unittest.TestCase):
    """Test fetch_arxiv_papers function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_paper = {
            'arxiv_id': '1234.5678',
            'title': 'Test Paper',
            'abstract': 'Test abstract',
            'authors': ['John Doe'],
            'published': '2023-01-15T00:00:00Z',
            'categories': ['cs.AI'],
            'primary_category': 'cs.AI'
        }
        self.module_path = os.path.join(project_root, 'scripts', '1_fetch_data.py')
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_fetch_arxiv_papers_defaults(self, mock_file):
        """Test that fetch_arxiv_papers uses defaults from config"""
        fd = load_module('fetch_data', self.module_path)
        with patch.object(fd, 'fetch_papers_by_category') as mock_fetch:
            mock_fetch.return_value = [self.sample_paper]
            result = fd.fetch_arxiv_papers()
            self.assertTrue(mock_fetch.call_count >= 1)
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_fetch_arxiv_papers_custom_params(self, mock_file):
        """Test fetch_arxiv_papers with custom parameters"""
        fd = load_module('fetch_data', self.module_path)
        with patch.object(fd, 'fetch_papers_by_category') as mock_fetch:
            mock_fetch.return_value = [self.sample_paper]
            result = fd.fetch_arxiv_papers(categories=['cs.CV'], start_year=2022)
            mock_fetch.assert_called_with('cs.CV', 2022)
            self.assertEqual(len(result), 1)
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_fetch_arxiv_papers_deduplication_across_categories(self, mock_file):
        """Test that fetch_arxiv_papers removes duplicates across categories"""
        fd = load_module('fetch_data', self.module_path)
        with patch.object(fd, 'fetch_papers_by_category') as mock_fetch:
            paper1 = {**self.sample_paper, 'arxiv_id': '1234.5678'}
            paper2 = {**self.sample_paper, 'arxiv_id': '1234.5678'}
            paper3 = {**self.sample_paper, 'arxiv_id': '9876.5432'}
            
            mock_fetch.side_effect = [[paper1], [paper2, paper3]]
            result = fd.fetch_arxiv_papers(categories=['cs.AI', 'cs.LG'], start_year=2020)
            
            self.assertEqual(len(result), 2)
            arxiv_ids = [p['arxiv_id'] for p in result]
            self.assertEqual(len(set(arxiv_ids)), 2)
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_fetch_arxiv_papers_empty_results(self, mock_file):
        """Test handling when no papers are found"""
        fd = load_module('fetch_data', self.module_path)
        with patch.object(fd, 'fetch_papers_by_category') as mock_fetch:
            mock_fetch.return_value = []
            result = fd.fetch_arxiv_papers(categories=['cs.AI'], start_year=2020)
            self.assertEqual(len(result), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.module_path = os.path.join(project_root, 'scripts', '1_fetch_data.py')
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_empty_year_range_future_year(self, mock_file):
        """Test when start_year is in the future"""
        with patch('arxiv.Client') as mock_client_class, \
             patch('arxiv.Search') as mock_search, \
             patch('datetime.datetime') as mock_datetime:
            
            fd = load_module('fetch_data', self.module_path)
            
            mock_datetime.now.return_value.year = 2023
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.results.return_value = iter([])
            mock_search.return_value = Mock()
            
            result = fd.fetch_papers_by_category('cs.AI', 2024)
            
            self.assertEqual(len(result), 0)
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_none_categories_handling(self, mock_file):
        """Test that None categories uses defaults"""
        fd = load_module('fetch_data', self.module_path)
        with patch.object(fd, 'fetch_papers_by_category') as mock_fetch:
            mock_fetch.return_value = []
            result = fd.fetch_arxiv_papers(categories=None, start_year=2020)
            self.assertTrue(mock_fetch.call_count > 0)
    
    @patch('builtins.open', new_callable=mock_open, read_data=mock_config_json)
    def test_none_start_year_handling(self, mock_file):
        """Test that None start_year uses defaults"""
        fd = load_module('fetch_data', self.module_path)
        with patch.object(fd, 'fetch_papers_by_category') as mock_fetch:
            mock_fetch.return_value = []
            result = fd.fetch_arxiv_papers(categories=['cs.AI'], start_year=None)
            mock_fetch.assert_called_with('cs.AI', 2020)


if __name__ == '__main__':
    unittest.main()
