import pytest
import pandas as pd
from .nl_processor import process_nl_query
from .analytics import execute_analytics
from .models import AnalyticsRequest

@pytest.fixture
def superstore_df():
    data = {
        'Row ID': [1, 2, 3, 4, 5, 6, 7],
        'Order Date': ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-10', '2023-03-05', '2023-03-20', '2023-04-01'],
        'Category': ['Furniture', 'Technology', 'Furniture', 'Office Supplies', 'Technology', 'Furniture', 'Technology'],
        'State': ['California', 'California', 'New York', 'Texas', 'California', 'Texas', 'California'],
        'Region': ['West', 'West', 'East', 'Central', 'West', 'Central', 'West'],
        'Product': ['Bookcase', 'Phone', 'Chair', 'Binder', 'Laptop', 'Table', 'Tablet'],
        'Sales': [300.0, 700.0, 200.0, 50.0, 1200.0, 450.0, 500.0],
        'Profit': [40.0, 150.0, 25.0, 10.0, 300.0, 60.0, 90.0]
    }
    return pd.DataFrame(data)

def test_sum_of_sales_for_furniture(superstore_df):
    """Bug 1 fix: Sum of Sales for Furniture must correctly apply the Furniture filter."""
    res = process_nl_query(superstore_df, "test.csv", "What is the total sales amount for the Furniture category?", None)
    expected_sum = superstore_df[superstore_df['Category'] == 'Furniture']['Sales'].sum() # 300 + 200 + 450 = 950.0
    assert res.result == expected_sum
    assert res.result == 950.0

def test_category_with_highest_sales(superstore_df):
    """Bug fix: Category with highest sales returns top performer, not Row ID."""
    res = process_nl_query(superstore_df, "test.csv", "Which category has the highest sales?", None)
    assert isinstance(res.result, dict)
    assert res.result.get('entity') == 'Technology' # 700 + 1200 + 500 = 2400.0

def test_major_category_sold_in_california(superstore_df):
    """Bug 2 fix: Major category sold in California filters by State=California and groups by Category."""
    res = process_nl_query(superstore_df, "test.csv", "What is the major category sold in California?", None)
    assert isinstance(res.result, dict)
    assert res.result.get('entity') == 'Technology' # in CA: Furniture=300, Technology=2400

def test_regions_in_california_distinct_count(superstore_df):
    """Bug 3 fix: How many regions were there in California counts distinct regions in CA instead of total rows."""
    res = process_nl_query(superstore_df, "test.csv", "How many regions were there in California?", None)
    assert isinstance(res.result, dict)
    assert res.result.get('count') == 1
    assert 'West' in res.result.get('unique_values', [])

def test_unique_region_count(superstore_df):
    """Counts unique regions."""
    res = process_nl_query(superstore_df, "test.csv", "How many unique regions are there?", None)
    expected_unique = superstore_df['Region'].nunique() # 3: West, East, Central
    assert isinstance(res.result, dict)
    assert res.result.get('count') == expected_unique
    assert res.result.get('count') == 3

def test_sales_filtered_by_california(superstore_df):
    """Sales filtered by State=California."""
    res = process_nl_query(superstore_df, "test.csv", "What is the total sales in California?", None)
    expected_ca_sales = superstore_df[superstore_df['State'] == 'California']['Sales'].sum() # 300 + 700 + 1200 + 500 = 2700.0
    assert res.result == expected_ca_sales
    assert res.result == 2700.0

def test_average_profit_by_region(superstore_df):
    """Average profit grouped by region."""
    res = process_nl_query(superstore_df, "test.csv", "What is the average profit by region?", None)
    assert isinstance(res.result, dict)
    expected_east = superstore_df[superstore_df['Region'] == 'East']['Profit'].mean()
    assert res.result.get('East') == expected_east

def test_invalid_or_ambiguous_question(superstore_df):
    """Gracefully handles completely irrelevant question."""
    res = process_nl_query(superstore_df, "test.csv", "Can you bake me a chocolate cake?", None)
    assert res.interpreted_as == "unknown" or res.result is None

def test_empty_filter_results(superstore_df):
    """Handles category or filter value that has 0 matching rows."""
    req = AnalyticsRequest(
        filename="test.csv",
        operation="sum",
        column="Sales",
        filter_column="Category",
        filter_value="NonExistentCategory"
    )
    res = execute_analytics(superstore_df, req)
    assert res.result == 0
    assert "No matching records found" in res.explanation
