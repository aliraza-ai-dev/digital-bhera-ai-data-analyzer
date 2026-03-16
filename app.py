from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI()

# Folders
UPLOAD_FOLDER = "static/uploads"
CHART_FOLDER = "charts"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/charts", StaticFiles(directory="charts"), name="charts")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/projects", response_class=HTMLResponse)
async def projects(request: Request):
    return templates.TemplateResponse("projects.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read CSV
    df = pd.read_csv(file_path)

    # Basic analysis
    num_rows, num_cols = df.shape
    columns = df.columns.tolist()
    preview = df.head().to_html(classes="table table-striped", index=False)
    missing_values = df.isnull().sum().to_dict()

    # Numeric statistics
    numeric_stats = df.describe().to_html(classes="table table-striped")

    # Numeric and categorical columns
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Categorical summary
    categorical_summary = {}
    for col in categorical_columns[:5]:
        categorical_summary[col] = df[col].value_counts().head(5).to_dict()

    # AI Insights
    insights = []
    insights.append(f"The dataset contains {num_rows} rows and {num_cols} columns.")

    total_missing = int(df.isnull().sum().sum())
    insights.append(f"There are {total_missing} missing values in the dataset.")

    if numeric_columns:
        for col in numeric_columns[:3]:
            avg = round(df[col].mean(), 2)
            insights.append(f"The average value of '{col}' is {avg}.")

        highest_missing_col = df.isnull().sum().idxmax()
        highest_missing_val = int(df.isnull().sum().max())
        if highest_missing_val > 0:
            insights.append(
                f"The column '{highest_missing_col}' has the highest missing values ({highest_missing_val})."
            )

        max_std_col = df[numeric_columns].std().idxmax()
        max_std_val = round(df[numeric_columns].std().max(), 2)
        insights.append(
            f"The column '{max_std_col}' shows the highest spread with standard deviation {max_std_val}."
        )

    if categorical_columns:
        first_cat = categorical_columns[0]
        top_category = df[first_cat].mode(dropna=True)
        if not top_category.empty:
            insights.append(
                f"The most frequent category in '{first_cat}' is '{top_category.iloc[0]}'."
            )

    # Correlation insight
    corr_matrix = df.corr(numeric_only=True)
    if not corr_matrix.empty and len(corr_matrix.columns) > 1:
        corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1]

        if len(corr_pairs) > 0:
            top_pair = corr_pairs.index[0]
            corr_value = round(corr_pairs.iloc[0], 2)
            insights.append(
                f"The strongest relationship is between '{top_pair[0]}' and '{top_pair[1]}' with correlation {corr_value}."
            )

    # Histogram
    chart_file = None
    if numeric_columns:
        first_numeric_col = numeric_columns[0]

        plt.figure(figsize=(8, 5))
        df[first_numeric_col].dropna().hist(bins=20)
        plt.title(f"Distribution of {first_numeric_col}")
        plt.xlabel(first_numeric_col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        chart_file = "chart.png"
        chart_path = os.path.join(CHART_FOLDER, chart_file)
        plt.savefig(chart_path)
        plt.close()

    # Heatmap
    heatmap_file = None
    if not corr_matrix.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()

        heatmap_file = "heatmap.png"
        heatmap_path = os.path.join(CHART_FOLDER, heatmap_file)
        plt.savefig(heatmap_path)
        plt.close()

    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": file.filename,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": columns,
        "preview": preview,
        "missing_values": missing_values,
        "numeric_stats": numeric_stats,
        "categorical_summary": categorical_summary,
        "insights": insights,
        "chart_file": chart_file,
        "heatmap_file": heatmap_file
    })