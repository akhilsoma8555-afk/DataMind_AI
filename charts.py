import pandas as pd
import json
import plotly.express as px
from typing import Dict, Any, Optional
from fastapi import HTTPException

PALETTE = [
    '#4F46E5', '#0284C7', '#0D9488', '#D97706', '#7C3AED', '#E11D48', '#475569'
]

def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_column: Optional[str] = None,
    group_by: Optional[str] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    if df.empty:
        raise HTTPException(status_code=400, detail="Empty data provided for chart generation")
        
    if x_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column not found: {x_column}")
        
    if y_column and y_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column not found: {y_column}")
        
    try:
        c_type = (chart_type or "bar").lower().strip()
        fig = None

        if c_type == "bar":
            if group_by and group_by in df.columns:
                agg_df = df.groupby([x_column, group_by])[y_column].sum().reset_index() if y_column else df.groupby([x_column, group_by]).size().reset_index(name="Count")
                y_col = y_column if y_column else "Count"
                fig = px.bar(agg_df, x=x_column, y=y_col, color=group_by, title=title, template="plotly_white", color_discrete_sequence=PALETTE)
            else:
                agg_df = df.groupby(x_column)[y_column].sum().reset_index() if y_column else df[x_column].value_counts().reset_index()
                agg_df.columns = [x_column, y_column or "Count"]
                fig = px.bar(agg_df, x=x_column, y=y_column or "Count", title=title, template="plotly_white", color_discrete_sequence=PALETTE)

        elif c_type in ["horizontal_bar", "barh", "ranked"]:
            agg_df = df.groupby(x_column)[y_column].sum().reset_index() if y_column else df[x_column].value_counts().reset_index()
            agg_df.columns = [x_column, y_column or "Count"]
            agg_df = agg_df.sort_values(by=y_column or "Count", ascending=True).tail(15)
            fig = px.bar(agg_df, x=y_column or "Count", y=x_column, orientation="h", title=title, template="plotly_white", color_discrete_sequence=PALETTE)

        elif c_type == "line":
            if group_by and group_by in df.columns:
                agg_df = df.groupby([x_column, group_by])[y_column].sum().reset_index()
                fig = px.line(agg_df, x=x_column, y=y_column, color=group_by, title=title, template="plotly_white", color_discrete_sequence=PALETTE)
            else:
                agg_df = df.groupby(x_column)[y_column].sum().reset_index()
                fig = px.line(agg_df, x=x_column, y=y_column, title=title, template="plotly_white", color_discrete_sequence=PALETTE)

        elif c_type in ["pie", "doughnut"]:
            agg_df = df.groupby(x_column)[y_column].sum().reset_index() if y_column else df[x_column].value_counts().reset_index()
            agg_df.columns = [x_column, y_column or "Count"]
            fig = px.pie(agg_df, names=x_column, values=y_column or "Count", title=title, template="plotly_white", hole=0.4, color_discrete_sequence=PALETTE)

        elif c_type in ["histogram", "distribution"]:
            fig = px.histogram(df, x=x_column, nbins=30, title=title or f"Distribution of {x_column}", template="plotly_white", color_discrete_sequence=PALETTE)

        elif c_type in ["scatter", "relationship"]:
            if not y_column:
                raise HTTPException(status_code=400, detail="Scatter plot requires both x_column and y_column")
            color_arg = group_by if group_by and group_by in df.columns else None
            fig = px.scatter(df, x=x_column, y=y_column, color=color_arg, title=title or f"{y_column} vs {x_column}", template="plotly_white", color_discrete_sequence=PALETTE)

        else:
            # Default to bar
            agg_df = df.groupby(x_column)[y_column].sum().reset_index() if y_column else df[x_column].value_counts().reset_index()
            agg_df.columns = [x_column, y_column or "Count"]
            fig = px.bar(agg_df, x=x_column, y=y_column or "Count", title=title, template="plotly_white", color_discrete_sequence=PALETTE)

        fig.update_layout(
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#172033"),
            margin=dict(l=55, r=25, t=45, b=50),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF"
        )
        return json.loads(fig.to_json())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")
