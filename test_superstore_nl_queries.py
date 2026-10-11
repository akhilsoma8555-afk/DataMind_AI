import pytest
import pandas as pd
from .nl_processor import process_nl_query
from .insights import generate_insights
from .analytics import execute_analytics
from .models import AnalyticsRequest

TRAIN_PATH = r"C:\Users\ASUS\Downloads\train.csv\train.csv"

@pytest.fixture(scope="module")
def train_df():
    df = pd.read_csv(TRAIN_PATH)
    return df

def test_1_what_is_the_total_sales(train_df):
    """TEST 1: What is the total sales? -> sum of Sales"""
    res = process_nl_query(train_df, "train.csv", "What is the total sales?", None)
    expected_sum = float(train_df["Sales"].sum())
    assert res.result is not None
    assert round(res.result, 2) == round(expected_sum, 2)
    assert round(res.result, 2) == 2261536.78

def test_2_different_product_categories(train_df):
    """TEST 2: What are the different product categories? -> list_unique on Category"""
    res = process_nl_query(train_df, "train.csv", "What are the different product categories?", None)
    expected_categories = set(train_df["Category"].dropna().unique())
    assert res.result is not None
    if isinstance(res.result, dict):
        vals = set(res.result.get("unique_values", res.result.get("values", [])))
    elif isinstance(res.result, list):
        vals = set(res.result)
    else:
        pytest.fail(f"Unexpected result format: {res.result}")
    assert vals == expected_categories
    assert "Furniture" in vals
    assert "Office Supplies" in vals
    assert "Technology" in vals

def test_3_different_shipping_modes(train_df):
    """TEST 3: What are the different shipping modes? -> list_unique on Ship Mode"""
    res = process_nl_query(train_df, "train.csv", "What are the different shipping modes?", None)
    expected_modes = set(train_df["Ship Mode"].dropna().unique())
    assert res.result is not None
    if isinstance(res.result, dict):
        vals = set(res.result.get("unique_values", res.result.get("values", [])))
    elif isinstance(res.result, list):
        vals = set(res.result)
    else:
        pytest.fail(f"Unexpected result format: {res.result}")
    assert vals == expected_modes
    assert "Standard Class" in vals
    assert "Second Class" in vals
    assert "First Class" in vals
    assert "Same Day" in vals

def test_4_how_many_unique_products(train_df):
    """TEST 4: How many unique products are there? -> count_unique (1861 by Product ID)"""
    res = process_nl_query(train_df, "train.csv", "How many unique products are there?", None)
    expected_count = int(train_df["Product ID"].nunique())
    assert isinstance(res.result, dict)
    assert res.result.get("count") == expected_count
    assert expected_count == 1861

def test_5_top_5_customers_by_total_sales(train_df):
    """TEST 5: Who are the top 5 customers by total sales? -> top_n on Customer Name"""
    res = process_nl_query(train_df, "train.csv", "Who are the top 5 customers by total sales?", None)
    top5_expected = train_df.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(5)
    assert isinstance(res.result, dict)
    ranking = res.result.get("ranking", {})
    assert len(ranking) == 5
    for name, val in top5_expected.items():
        assert name in ranking
        assert round(ranking[name], 2) == round(val, 2)
    assert list(ranking.keys())[0] == top5_expected.index[0]
    assert list(ranking.keys())[0] == "Sean Miller"

def test_6_lowest_sales_sub_categories(train_df):
    """TEST 6: Which sub-categories have the lowest total sales? -> bottom_n on Sub-Category"""
    res = process_nl_query(train_df, "train.csv", "Which sub-categories have the lowest total sales?", None)
    bottom_expected = train_df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=True).head(5)
    assert isinstance(res.result, dict)
    ranking = res.result.get("ranking", {})
    assert len(ranking) >= 3
    lowest_name = bottom_expected.index[0]
    assert lowest_name in ranking
    assert round(ranking[lowest_name], 2) == round(bottom_expected.iloc[0], 2)
    assert lowest_name == "Fasteners"

def test_7_compare_sales_between_shipping_modes(train_df):
    """TEST 7: Compare sales between different shipping modes. -> group_aggregate on Ship Mode"""
    res = process_nl_query(train_df, "train.csv", "Compare sales between different shipping modes.", None)
    expected = train_df.groupby("Ship Mode")["Sales"].sum().round(2).to_dict()
    assert isinstance(res.result, dict)
    for mode, val in expected.items():
        assert mode in res.result
        assert round(res.result[mode], 2) == round(val, 2)

def test_8_total_sales_furniture_california(train_df):
    """TEST 8: What is the total sales for Furniture in California? -> sum with multiple filters"""
    res = process_nl_query(train_df, "train.csv", "What is the total sales for Furniture in California?", None)
    expected = float(train_df[(train_df["Category"] == "Furniture") & (train_df["State"] == "California")]["Sales"].sum())
    assert res.result is not None
    assert round(res.result, 2) == round(expected, 2)
    assert round(res.result, 2) == 152216.54

def test_9_show_monthly_sales_trend(train_df):
    """TEST 9: Show the monthly sales trend. -> time_series"""
    res = process_nl_query(train_df, "train.csv", "Show the monthly sales trend.", None)
    assert isinstance(res.result, dict)
    assert res.chart_data is not None
    assert res.chart_data.get("type") == "line"
    months = list(res.result.keys())
    assert len(months) >= 12
    assert months == sorted(months)

def test_10_generate_business_performance_report(train_df):
    """TEST 10: Generate a business performance report."""
    res = process_nl_query(train_df, "train.csv", "Generate a business performance report.", None)
    assert isinstance(res.result, dict)
    assert "report_title" in res.result or "metrics" in res.result
    assert "kpis" in res.result or "metrics" in res.result
    assert "recommendations" in res.result

def test_11_analytics_monthly_trend_execution(train_df):
    """TEST 11: Analytics Trend Analysis with Date Column = Order Date, Value Column = Sales, Interval = Month"""
    req = AnalyticsRequest(
        filename="train.csv",
        operation="trend",
        date_column="Order Date",
        value_column="Sales",
        interval="month"
    )
    res = execute_analytics(train_df, req)
    assert res.operation == "trend"
    assert isinstance(res.result, dict)
    assert len(res.result) == 48  # 4 years of monthly data
    
    # Verify chart data
    assert res.chart_data is not None
    assert res.chart_data["type"] == "line"
    assert len(res.chart_data["x"]) == 48
    assert len(res.chart_data["y"]) == 48
    assert res.chart_data["title"] == "Monthly Trend of Sales"
    # Ensure chronological sort
    assert res.chart_data["x"] == sorted(res.chart_data["x"])
    assert res.chart_data["x"][0] == "2015-01"
    assert res.chart_data["x"][-1] == "2018-12"

def test_12_analytics_validation_errors(train_df):
    """TEST 12: Validate error reporting on non-numeric value column or missing dates"""
    from fastapi import HTTPException
    # Non-numeric value column error
    req_invalid_val = AnalyticsRequest(
        filename="train.csv",
        operation="trend",
        date_column="Order Date",
        value_column="Customer Name",
        interval="month"
    )
    with pytest.raises(HTTPException) as exc_info:
        execute_analytics(train_df, req_invalid_val)
    assert "non-numeric" in exc_info.value.detail

def test_13_comprehensive_insights(train_df):
    """TEST 13: Full dataset-driven insights on train.csv"""
    insights_res = generate_insights(train_df, None)
    assert insights_res.overview.row_count == 9800
    assert insights_res.overview.column_count == 18
    assert len(insights_res.kpis) >= 3
    assert len(insights_res.performers) >= 2
    assert len(insights_res.charts) >= 1
    assert len(insights_res.recommendations) >= 2
