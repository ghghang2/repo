"""
Unit tests for search_engine.perform_search using a mocked browser tool.
"""
import unittest
from unittest.mock import patch

# Import module
from src import search_engine

class TestSearchEngine(unittest.TestCase):

    def setUp(self):
        # Provide a simple HTML snippet that mimics DuckDuckGo results
        self.html = """
        <div class="result">
            <a class="result__a" href="https://example.com">Example Title</a>
            <a class="result__a" href="https://example.com">https://example.com</a>
            <a class="result__snippet">This is a snippet.</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://example.org">Another Title</a>
            <a class="result__a" href="https://example.org">https://example.org</a>
            <a class="result__snippet">Another snippet.</a>
        </div>
        """

    @patch('src.search_engine.browser')
    def test_perform_search(self, mock_browser):
        # Mock the browser tool to return our HTML
        mock_browser.return_value = {'text': self.html}
        results = search_engine.perform_search("test query", num_results=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['title'], 'Example Title')
        self.assertEqual(results[0]['url'], 'https://example.com')
        self.assertEqual(results[0]['snippet'], 'This is a snippet.')
        self.assertEqual(results[1]['title'], 'Another Title')

if __name__ == '__main__':
    unittest.main()
"""
