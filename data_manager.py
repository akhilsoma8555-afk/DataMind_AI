import pandas as pd
import io
import random
from datetime import datetime, timedelta
from fastapi import HTTPException, UploadFile
from typing import Dict, List, Optional
from models import UploadResponse, ColumnInfo

_cache: Dict[str, pd.DataFrame] = {}
_raw_cache: Dict[str, pd.DataFrame] = {}
_metadata: Dict[str, Dict[str, Any]] = {}
_active_dataset: Optional[str] = None

def set_active_dataset(filename: str) -> None:
    global _active_dataset
    if filename in _cache:
        _active_dataset = filename
    elif filename in _raw_cache:
        _active_dataset = filename
    else:
        raise HTTPException(status_code=404, detail=f"Dataset '{filename}' is not registered in the system.")

def get_active_dataset() -> Optional[str]:
    global _active_dataset
    if _active_dataset and _active_dataset in _cache:
        return _active_dataset
    if _cache:
        return list(_cache.keys())[-1]
    return None

def list_datasets() -> List[Dict[str, Any]]:
    global _active_dataset
    active = get_active_dataset()
    result = []
    for fname, df in _cache.items():
        meta = _metadata.get(fname, {})
        result.append({
            "filename": fname,
            "row_count": len(df),
            "column_count": len(df.columns),
            "is_active": (fname == active),
            "upload_time": meta.get("upload_time", datetime.now().strftime("%Y-%m-%d %H:%M"))
        })
    return result

def update_dataframe(filename: str, df: pd.DataFrame) -> None:
    _cache[filename] = df
    if filename in _metadata:
        _metadata[filename]["row_count"] = len(df)
        _metadata[filename]["column_count"] = len(df.columns)

def get_raw_dataframe(filename: Optional[str] = None) -> pd.DataFrame:
    act = filename or get_active_dataset()
    if act and act in _raw_cache:
        return _raw_cache[act].copy()
    if act and act in _cache:
        return _cache[act].copy()
    raise HTTPException(status_code=404, detail="Dataset not found")

def process_dataframe(df: pd.DataFrame, filename: str) -> UploadResponse:
    global _active_dataset
    # Preserve raw untouched copy before any processing
    raw_copy = df.copy()
    _raw_cache[filename] = raw_copy

    # Compute missing values BEFORE filling
    missing_values = df.isna().sum().to_dict()

    # Work on a copy for analytical operations where nulls should be gracefully handled
    working_df = df.copy()
    for col in working_df.select_dtypes(include=['number']).columns:
        working_df[col] = working_df[col].fillna(0)
    for col in working_df.select_dtypes(include=['object', 'category']).columns:
        working_df[col] = working_df[col].fillna('Unknown')
    
    _cache[filename] = working_df
    _active_dataset = filename
    _metadata[filename] = {
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "row_count": len(df),
        "column_count": len(df.columns)
    }
    
    columns = [ColumnInfo(name=col, dtype=str(raw_copy[col].dtype)) for col in raw_copy.columns]
    preview = raw_copy.head(10).to_dict(orient="records")
    # Clean preview NaNs for JSON serialization
    for row in preview:
        for k, v in row.items():
            if pd.isna(v):
                row[k] = None
    suggested = generate_suggested_questions(raw_copy, filename)
    
    return UploadResponse(
        filename=filename,
        row_count=len(raw_copy),
        column_count=len(raw_copy.columns),
        columns=columns,
        preview=preview,
        missing_values=missing_values,
        suggested_questions=suggested
    )

async def save_upload(file: UploadFile) -> UploadResponse:
    fname = (file.filename or "").lower()
    if not (fname.endswith('.csv') or fname.endswith('.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx) files are supported")
    
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
        
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        
    if len(df) > 100000:
        raise HTTPException(status_code=400, detail="File exceeds 100,000 rows limit")
        
    return process_dataframe(df, file.filename)

def get_dataframe(filename: Optional[str] = None) -> pd.DataFrame:
    global _active_dataset
    # If a specific filename is requested and present
    if filename and filename != "active" and filename in _cache:
        return _cache[filename]
    
    # Fallback to active dataset
    if _active_dataset and _active_dataset in _cache:
        return _cache[_active_dataset]
        
    # Fallback to most recent cache entry
    if _cache:
        active_key = list(_cache.keys())[-1]
        _active_dataset = active_key
        return _cache[active_key]
        
    raise HTTPException(status_code=404, detail="No dataset loaded. Please upload a CSV or Excel file first.")

def generate_suggested_questions(df: pd.DataFrame, filename: str) -> List[str]:
    """Dynamically generate suggested business questions based solely on the actual columns and data of df."""
    cols = list(df.columns)
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if not any(k in c.lower() for k in ['row id', 'id', 'postal', 'zip', 'code'])]
    cat_cols = list(df.select_dtypes(include=['object', 'category', 'string']).columns)
    
    questions = []
    
    # 1. Identify primary metric (e.g. Price, Sales, Profit, Mileage, Quantity)
    price_col = next((c for c in num_cols if 'price' in c.lower() or 'cost' in c.lower()), None)
    sales_col = next((c for c in num_cols if 'sale' in c.lower() or 'revenue' in c.lower()), None)
    profit_col = next((c for c in num_cols if 'profit' in c.lower() or 'margin' in c.lower()), None)
    mileage_col = next((c for c in num_cols if 'mileage' in c.lower() or 'distance' in c.lower()), None)
    
    metric_col = None
    metric_type = "sum"
    
    if price_col:
        metric_col = price_col
        metric_type = "avg"
        questions.append(f"What is the average {price_col}?")
    elif sales_col:
        metric_col = sales_col
        metric_type = "sum"
        questions.append(f"What is the total {sales_col}?")
    elif profit_col:
        metric_col = profit_col
        metric_type = "sum"
        questions.append(f"What is the total {profit_col}?")
    elif num_cols:
        metric_col = num_cols[0]
        metric_type = "avg"
        questions.append(f"What is the average {metric_col}?")

    # 2. Identify primary grouping entity (e.g. Car Make, Category, Brand, Manufacturer)
    primary_entity = next((c for c in cat_cols if any(k in c.lower() for k in ['make', 'manufacturer', 'brand', 'category'])), None)
    if not primary_entity and cat_cols:
        primary_entity = cat_cols[0]

    if primary_entity and metric_col:
        if metric_type == "avg":
            questions.append(f"Which {primary_entity} has the highest average {metric_col}?")
        else:
            questions.append(f"Which {primary_entity} has the highest total {metric_col}?")

    # 3. Identify secondary entity or model (e.g. Car Model, Sub-Category, Product, Item)
    secondary_entity = next((c for c in cat_cols if any(k in c.lower() for k in ['model', 'sub-category', 'subcategory', 'product', 'item'])), None)
    if not secondary_entity:
        for c in cat_cols:
            if c != primary_entity:
                secondary_entity = c
                break

    if secondary_entity:
        label = secondary_entity if secondary_entity.endswith('s') else f"{secondary_entity}s"
        questions.append(f"How many unique {label} are there?")

    # 4. Dimension comparison (e.g. Fuel Type, Transmission, Region, Ship Mode)
    compare_dim = next((c for c in cat_cols if any(k in c.lower() for k in ['fuel', 'transmission', 'type', 'region', 'ship mode', 'segment'])), None)
    if not compare_dim:
        for c in cat_cols:
            if c not in [primary_entity, secondary_entity]:
                compare_dim = c
                break

    if compare_dim and metric_col:
        if metric_type == "avg":
            questions.append(f"Compare average {metric_col} by {compare_dim}.")
        else:
            questions.append(f"Compare {metric_col} between different {compare_dim}s.")

    # 5. Temporal / Chronological analysis (Year or Date column)
    year_col = next((c for c in cols if 'year' in c.lower()), None)
    date_col = next((c for c in cols if 'date' in c.lower() or 'time' in c.lower()), None)
    
    if year_col and metric_col:
        questions.append(f"How has the average {metric_col} changed by {year_col}?")
    elif date_col and metric_col:
        questions.append(f"Show the monthly {metric_col.lower()} trend.")

    # 6. Distinct values question
    list_cand = compare_dim or primary_entity or (cat_cols[0] if cat_cols else None)
    if list_cand:
        label = list_cand if list_cand.endswith('s') else f"{list_cand}s"
        questions.append(f"What are the different {label}?")

    # 7. Additional interesting breakdown (Mileage or Top N items)
    if mileage_col and primary_entity:
        questions.append(f"What is the average {mileage_col} by {primary_entity}?")
    elif primary_entity and metric_col:
        label = primary_entity if primary_entity.endswith('s') else f"{primary_entity}s"
        questions.append(f"What are the top 5 {label} by {metric_col}?")

    # 8. Report query
    questions.append("Generate a business performance report.")

    seen = set()
    unique_questions = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)
    return unique_questions[:8]

def get_sample_data() -> UploadResponse:
    filename = "sample_data.csv"
    if filename in _cache:
        return process_dataframe(_cache[filename], filename)
        
    # Generate realistic business data
    dates = [datetime(2023, 1, 1) + timedelta(days=i*5) for i in range(200)]
    products = ['Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Accessories']
    categories = ['Electronics', 'Accessories']
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'Date': dates,
        'Product': [random.choice(products) for _ in range(200)],
        'Category': [random.choice(categories) for _ in range(200)],
        'Region': [random.choice(regions) for _ in range(200)],
        'Quantity': [random.randint(1, 50) for _ in range(200)],
    }
    
    df = pd.DataFrame(data)
    df['Sales'] = df['Quantity'] * df['Product'].map({'Laptop': 1000, 'Smartphone': 600, 'Tablet': 400, 'Monitor': 300, 'Accessories': 50}) * (1 + (pd.Series([random.uniform(-0.1, 0.1) for _ in range(200)])))
    df['Profit'] = df['Sales'] * df['Product'].map({'Laptop': 0.15, 'Smartphone': 0.20, 'Tablet': 0.25, 'Monitor': 0.10, 'Accessories': 0.40})
    
    return process_dataframe(df, filename)
