# Polars Cheatsheet

A comprehensive guide to [Polars](https://pola.rs/), a localized (single-machine) high-performance DataFrame library for Python, built on Rust and Apache Arrow.

## 🐍 Pandas vs. 🐻‍❄️ Polars: Key Differences & Syntax Mapping

| Feature | Pandas (`pd`) | Polars (`pl`) | Key Difference |
| :--- | :--- | :--- | :--- |
| **Index** | Has a row Index (often implicit). | **No Index.** Row numbers are implicit. | Polars doesn't use indexes for lookups; it uses physical memory offsets. |
| **Missing Values** | `NaN` (float), `None` (object), `pd.NA`. | **`null`** (applies to all types). | Consistent missing value handling across all data types. |
| **Selection** | `df[['a', 'b']]` | `df.select(['a', 'b'])` | Polars discourages `[]` for col selection in favor of `.select()`. |
| **Filtering** | `df[df['a'] > 5]` | `df.filter(pl.col('a') > 5)` | Uses `.filter()` with expressions. |
| **New Column** | `df['sum'] = df['a'] + df['b']` | `df.with_columns((pl.col('a') + pl.col('b')).alias('sum'))` | **Immutable.** `.with_columns()` returns a new DataFrame. |
| **Grouping** | `df.groupby('a').agg({'b': 'sum'})` | `df.group_by('a').agg(pl.col('b').sum())` | `group_by` (v1.0+) vs `groupby`. Aggregation uses expressions. |
| **Sorting** | `df.sort_values('a')` | `df.sort('a')` | Simpler method name. |
| **Strings** | `df['s'].str.upper()` | `df.select(pl.col('s').str.to_uppercase())` | String methods under `.str` namespace in expressions. |
| **Unpivoting** | `df.melt(...)` | `df.unpivot(...)` | `unpivot` is preferred over `melt` in recent Polars versions. |
| **Execution** | Eager (mostly). | **Lazy & Eager.** | Polars emphasizes `LazyFrame` for query optimization. |

### 💡 Key Philosophy
1.  **Expressions are King**: Nearly everything in Polars (selection, filtering, aggregation, column creation) uses **Expressions** (`pl.col(...)`). They are lazy, composable, and optimizer-friendly.
2.  **No Index**: Use standard columns for identifiers. Valid index usage in Pandas (like `loc`) is often handled by `filter` in Polars.
3.  **Parallelization**: Polars automatically parallelizes operations utilizing all available cores without custom logic.
4.  **Method Chaining**: Writing almost everything as a chain of methods allows the Polars query optimizer to see the "whole picture," creating a more efficient execution plan than step-by-step Pandas execution.

---

## 🚀 Installation & Basics

```bash
pip install polars
# For improved performance on old CPUs or specific hardware, check documentation for optimized builds (e.g. polars-lts-cpu)
```

```python
import polars as pl
import pandas as pd

# From Dictionary
data = {"a": [1, 2, 3], "b": [4, 5, 6]}
df = pl.DataFrame(data)

# From Pandas
df_pd = pd.DataFrame(data)
df = pl.from_pandas(df_pd)
```

---

## 💾 Input / Output

| Format | Read | Write (Eager) | Write (Lazy) |
| :--- | :--- | :--- | :--- |
| **CSV** | `pl.read_csv("file.csv")` | `df.write_csv("file.csv")` | `df_lazy.sink_csv(...)` |
| **Parquet** | `pl.read_parquet("file.parquet")` | `df.write_parquet("file.parquet")` | `df_lazy.sink_parquet(...)` |
| **JSON** | `pl.read_json("file.json")` | `df.write_json("file.json")` | `df_lazy.sink_json(...)` |
| **Database** | `pl.read_database_uri(query, uri)` | `df.write_database(...)` | - |

**Lazy Scanning** (Does not load data into memory immediately):
```python
lf = pl.scan_csv("data.csv")
lf = pl.scan_parquet("data.parquet")
```

---

## 🔍 Selection & Viewing

```python
df.head(5)        # First 5 rows
df.tail(5)        # Last 5 rows
df.glimpse()      # Dense overview of data types and values (like R's glimpse)
df.schema         # Dict of column names to DataTypes
df.columns        # List of column names

# Select specific columns
df.select("a", "b")
df.select(["a", "b"])
df.select(pl.col("a"), pl.col("b"))

# Select with exclusion
df.select(pl.all().exclude("b"))

# Select by type
# Select by type using Selectors (cs)
import polars.selectors as cs
df.select(cs.numeric())          # All numeric columns
df.select(cs.string())           # All string columns
df.select(cs.by_name("res.*"))   # Columns matching regex
df.select(~cs.numeric())         # detailed negation (NOT numeric)
```

---

## 🔽 Filtering (Rows)

Polars uses `.filter()` with boolean expressions.

```python
# Simple filter
df.filter(pl.col("a") > 2)

# Multiple conditions (& for AND, | for OR)
df.filter((pl.col("a") > 1) & (pl.col("b") < 10))

# Is In / Is Null
df.filter(pl.col("a").is_in([1, 3, 5]))
df.filter(pl.col("a").is_null())
df.filter(pl.col("a").is_not_null())
```

---

## ⚡ Expressions: The Power House

Expressions (`pl.col(...)`) define transformations solely on columns. They can be used in `select`, `with_columns`, `filter`, `group_by`, etc.

### Core Arithmetic & Logic
```python
pl.col("a") + pl.col("b")
pl.col("a") * 2
(pl.col("a") > 0).alias("is_positive") # Rename result
pl.when(pl.col("a") > 5).then("High").otherwise("Low") # If-Else style
```

### String Operations (`.str`)
```python
pl.col("s").str.to_uppercase()
pl.col("s").str.contains("pattern")
pl.col("s").str.replace("b", "B")
pl.col("s").str.slice(0, 3) # Substring
pl.col("s").str.split(" ")
```

### Date & Time (`.dt`)
```python
pl.col("date").dt.year()
pl.col("date").dt.month()
pl.col("date").dt.strftime("%Y-%m-%d")
pl.col("date").dt.add_business_days(5)
```

### List / Array Operations (`.list`)
```python
pl.col("list_col").list.len()
pl.col("list_col").list.get(0)      # First element
pl.col("list_col").list.join(", ")  # Join elements to string
pl.col("list_col").list.explode()   # Unnest list to rows
```

---

## 🛠 Column Manipulation

**`with_columns`** is the primary method to add, modify, or compute columns. It runs operations in parallel.

```python
# Add/Update multiple columns
df = df.with_columns(
    (pl.col("a") * 2).alias("a_doubled"),
    (pl.col("b") / 10).alias("b_scaled"),
    pl.lit("constant_value").alias("const") # Literal value
)

# Casting types
df = df.with_columns(
    pl.col("a").cast(pl.Int32),
    pl.col("date_str").str.to_datetime("%Y-%m-%d")
)

# Sorting / Renaming / Dropping
df.sort("a", descending=True)
df.rename({"old_name": "new_name"})
df.drop("col_to_remove")
```

---

## 📊 Aggregation & Grouping

Use `group_by` (preferred over `groupby`) followed by `agg`.

```python
df.group_by("category").agg(
    pl.len().alias("count"),            # Count rows in group
    pl.col("val").sum().alias("sum"),   # Sum of 'val'
    pl.col("val").mean(),               # Mean of 'val'
    pl.col("val").max(),                # Max of 'val'
    pl.col("val").first(),              # First value
    pl.col("names").list()              # Collect values into a list
)
```

### Window Functions (`.over()`)
Compute aggregations *without* collapsing groups (like SQL window functions). Extremely powerful.

```python
# Add a column with group average alongside original data
df.with_columns(
    pl.col("val").mean().over("category").alias("cat_mean")
)

# Calculate difference from group mean
df.with_columns(
    (pl.col("val") - pl.col("val").mean().over("category")).alias("diff_from_mean")
)
```

---

## ⚡ Performance Tips: The Power of Method Chaining

In Pandas, you often assign intermediate results to variables. In Polars, **chaining methods** is critical for performance, especially in Lazy mode. It allows the Query Optimizer to:
1.  **Predicate Pushdown**: Apply filters *before* loading data or expensive operations.
2.  **Projection Pushdown**: Load only necessary columns.
3.  **Common Subexpression Elimination**: Avoid redundant calculations.

### Pandas Style (Avoid)
```python
# ❌ Intermediate steps block optimization and memory release
df1 = pl.read_csv("data.csv")
df2 = df1.filter(pl.col("status") == "active")
df3 = df2.select(["id", "val"])
result = df3.group_by("id").agg(pl.col("val").sum())
```

### Polars Style (Preferred)
```python
# ✅ Chaining allows Polars to optimize the entire query
result = (
    pl.scan_csv("data.csv")  # Lazy load
    .filter(pl.col("status") == "active")
    .select("id", "val")
    .group_by("id")
    .agg(pl.col("val").sum())
    .collect()
)
```

---

## 🔗 Joins & Combination

### Joins
```python
# Standard SQL-style joins: inner, left, outer, semi, anti, cross
df_a.join(df_b, on="id", how="inner")
df_a.join(df_b, left_on="a_id", right_on="b_id", how="left")
```

### Concatenation
```python
# Vertical (Stacking rows)
pl.concat([df1, df2], how="vertical") # Equivalent to vstack

# Horizontal (Stacking columns - MUST have same height)
pl.concat([df1, df2], how="horizontal") # Equivalent to hstack
```

---

## 🔄 Reshaping

| Operation | Method | Description |
| :--- | :--- | :--- |
| **Pivot** | `.pivot()` | Long -> Wide. |
| **Unpivot** | `.unpivot()` | Wide -> Long. (Formerly `melt`) |
| **Transpose** | `.transpose()` | Swap rows and columns. |

```python
# Pivot
df.pivot(on="year", index="country", values="gdp")

# Unpivot (Melt)
df.unpivot(index=["country"], on=["2020", "2021"], variable_name="year", value_name="gdp")
```

---

## 🚫 Missing Data Handling

```python
# Fill Nulls
df.with_columns(pl.col("a").fill_null(0))
df.with_columns(pl.col("a").fill_null(strategy="forward"))

# Drop Nulls
df.drop_nulls()               # Drop row if ANY column is null
df.drop_nulls(subset=["a"])   # Drop row if 'a' is null

# Replace NaNs (Floats only, distinct from Null)
df.with_columns(pl.col("val").fill_nan(0))
```

---

## 💤 Lazy API (Performance Booster)

The "Lazy" API records operations without executing them. When `collect()` is called, Polars builds a query plan, optimizes it (predicate pushdown, projection pushdown), and executes it.

```python
# 1. Start with a LazyFrame
lf = pl.scan_csv("huge_file.csv")

# 2. Chain operations (nothing happens yet)
q = (
    lf.filter(pl.col("val") > 100)
    .group_by("group")
    .agg(pl.col("val").mean())
)

# 3. Optimize & Execute
df_result = q.collect()  # Returns eager DataFrame

# Or Stream results (for larger-than-RAM datasets)
q.sink_parquet("output.parquet")
```

### Inspecting Plans
```python
q.explain() # Print the unoptimized query plan
q.explain(optimized=True) # Print the optimized plan
```

---

## ⚙️ Configuration

```python
# Adjust display settings
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(10)
pl.Config.set_fmt_str_lengths(50)
```
