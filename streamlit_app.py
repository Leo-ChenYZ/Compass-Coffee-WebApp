from collections import defaultdict
from pathlib import Path
import sqlite3

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from modeling import run_forecast


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title="Compass Coffee Reorder Dashboard",
    #page_icon=":shopping_bags:",  # This is an emoji shortcode. Could be a URL too.
)


# -----------------------------------------------------------------------------
# Declare some useful functions.


def connect_db():
    """Connects to the sqlite database."""

    DB_FILENAME = Path(__file__).parent / "inventory.db"
    db_already_exists = DB_FILENAME.exists()

    conn = sqlite3.connect(DB_FILENAME)
    db_was_just_created = not db_already_exists

    return conn, db_was_just_created


def initialize_data(conn):
    """Initializes the inventory table with some data."""
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT,
            price REAL,
            units_sold INTEGER,
            units_left INTEGER,
            cost_price REAL,
            reorder_point INTEGER,
            description TEXT
        )
        """
    )

    cursor.execute(
        """
        INSERT INTO inventory
            (item_name, price, units_sold, units_left, cost_price, reorder_point, description)
        VALUES
            -- Beverages
            ('Bottled Water (500ml)', 1.50, 115, 15, 0.80, 16, 'Hydrating bottled water'),
            ('Soda (355ml)', 2.00, 93, 8, 1.20, 10, 'Carbonated soft drink'),
            ('Energy Drink (250ml)', 2.50, 12, 18, 1.50, 8, 'High-caffeine energy drink'),
            ('Coffee (hot, large)', 2.75, 11, 14, 1.80, 5, 'Freshly brewed hot coffee'),
            ('Juice (200ml)', 2.25, 11, 9, 1.30, 5, 'Fruit juice blend'),

            -- Snacks
            ('Potato Chips (small)', 2.00, 34, 16, 1.00, 10, 'Salted and crispy potato chips'),
            ('Candy Bar', 1.50, 6, 19, 0.80, 15, 'Chocolate and candy bar'),
            ('Granola Bar', 2.25, 3, 12, 1.30, 8, 'Healthy and nutritious granola bar'),
            ('Cookies (pack of 6)', 2.50, 8, 8, 1.50, 5, 'Soft and chewy cookies'),
            ('Fruit Snack Pack', 1.75, 5, 10, 1.00, 8, 'Assortment of dried fruits and nuts'),

            -- Personal Care
            ('Toothpaste', 3.50, 1, 9, 2.00, 5, 'Minty toothpaste for oral hygiene'),
            ('Hand Sanitizer (small)', 2.00, 2, 13, 1.20, 8, 'Small sanitizer bottle for on-the-go'),
            ('Pain Relievers (pack)', 5.00, 1, 5, 3.00, 3, 'Over-the-counter pain relief medication'),
            ('Bandages (box)', 3.00, 0, 10, 2.00, 5, 'Box of adhesive bandages for minor cuts'),
            ('Sunscreen (small)', 5.50, 6, 5, 3.50, 3, 'Small bottle of sunscreen for sun protection'),

            -- Household
            ('Batteries (AA, pack of 4)', 4.00, 1, 5, 2.50, 3, 'Pack of 4 AA batteries'),
            ('Light Bulbs (LED, 2-pack)', 6.00, 3, 3, 4.00, 2, 'Energy-efficient LED light bulbs'),
            ('Trash Bags (small, 10-pack)', 3.00, 5, 10, 2.00, 5, 'Small trash bags for everyday use'),
            ('Paper Towels (single roll)', 2.50, 3, 8, 1.50, 5, 'Single roll of paper towels'),
            ('Multi-Surface Cleaner', 4.50, 2, 5, 3.00, 3, 'All-purpose cleaning spray'),

            -- Others
            ('Lottery Tickets', 2.00, 17, 20, 1.50, 10, 'Assorted lottery tickets'),
            ('Newspaper', 1.50, 22, 20, 1.00, 5, 'Daily newspaper')
        """
    )
    conn.commit()


def load_data(conn):
    """Loads the inventory data from the database."""
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM inventory")
        data = cursor.fetchall()
    except:
        return None

    df = pd.DataFrame(
        data,
        columns=[
            "id",
            "item_name",
            "price",
            "units_sold",
            "units_left",
            "cost_price",
            "reorder_point",
            "description",
        ],
    )

    return df


def update_data(conn, df, changes):
    """Updates the inventory data in the database."""
    cursor = conn.cursor()

    if changes["edited_rows"]:
        deltas = st.session_state.inventory_table["edited_rows"]
        rows = []

        for i, delta in deltas.items():
            row_dict = df.iloc[i].to_dict()
            row_dict.update(delta)
            rows.append(row_dict)

        cursor.executemany(
            """
            UPDATE inventory
            SET
                item_name = :item_name,
                price = :price,
                units_sold = :units_sold,
                units_left = :units_left,
                cost_price = :cost_price,
                reorder_point = :reorder_point,
                description = :description
            WHERE id = :id
            """,
            rows,
        )

    if changes["added_rows"]:
        cursor.executemany(
            """
            INSERT INTO inventory
                (id, item_name, price, units_sold, units_left, cost_price, reorder_point, description)
            VALUES
                (:id, :item_name, :price, :units_sold, :units_left, :cost_price, :reorder_point, :description)
            """,
            (defaultdict(lambda: None, row) for row in changes["added_rows"]),
        )

    if changes["deleted_rows"]:
        cursor.executemany(
            "DELETE FROM inventory WHERE id = :id",
            ({"id": int(df.loc[i, "id"])} for i in changes["deleted_rows"]),
        )

    conn.commit()


def apply_unit_conversion(predictions: pd.DataFrame, conversion_df: pd.DataFrame) -> pd.DataFrame:
    """Attach unit conversion info and compute converted forecast values."""
    conversion_df = conversion_df.rename(
        columns={"item": "Product", "Primary Purchase Unit": "Unit", "Factor": "Factor"}
    )

    output = predictions.merge(
        conversion_df[["Product", "Factor", "Unit"]],
        on="Product",
        how="left",
    )
    output["Factor"] = output["Factor"].fillna(1.0).astype(float)
    output["Unit"] = output["Unit"].fillna("base_unit")
    output["Predicted Amount (Next 7 Days)"] = (
        np.ceil(output["Predicted Amount (Next 7 Days)"] * output["Factor"])
        .fillna(0)
        .astype(int)
    )
    return output


# -----------------------------------------------------------------------------
# Draw the actual page, starting with the inventory table.

# Set the title that appears at the top of the page.
"""
# Compass Coffee Reorder Dashboard

**Welcome to Compass Coffee's reorder dashboard for store managers!**
Using recent sales data you upload, this page provides inventory stocking recommendations using Machine Learning. These are only suggestions, and the developers of this page are not responsible for any errors. For internal use only.
"""



# CSV file upload for demand prediction
st.subheader("Upload Recent Sales Data")
uploaded_file = st.file_uploader(
    "Upload a CSV file for demand prediction",
    type="csv",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

# Store location selector
st.subheader("Select Store Location")
store_location = st.selectbox(
    "Choose a store:",
    sorted([
        "Adams Morgan",
        "Dupont",
        "Navy Yard, Ballpark",
        "Ballston",
        "Fairfax",
        "Spring Valley",
        "Ballston West, Wilson & N. Glebe",
        "College Park",
        "Langston Boulevard, Drive Thru",
        "Clarendon",
        "Chinatown, 7th & F",
        "Rosslyn",
        "Shaw, 7th & P",
        "Navy Yard North, I & New Jersey",
        "West Falls Church, Drive Thru",
        "Franklin Park, 13th & K",
        "Penn Quarter, 11th & E",
        "Farragut, 18th & Eye",
        "Mount Vernon, 7th & New York",
        "14th Street, 14th & U",
        "Golden Triangle, 17th & H",
        "McPherson, 14th & Eye",
        "Georgetown",
        "North Shaw, 8th & Florida",
        "Metro Center, 13th and F"
    ]),
)

# Prediction section
st.subheader("Generate Demand Predictions")
st.write("Click the button below to generate demand predictions for the next 7 days.")

# Initialize persistent session state keys
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "order_results" not in st.session_state:
    st.session_state.order_results = None

# Generate predictions and persist in session state
if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
    try:
        data_path = Path(__file__).parent / "mock_preprocessed_data.csv"
        df = pd.read_csv(data_path)

        conversion_path = Path(__file__).parent / "item_unit_conversions.csv"
        conversion_df = pd.read_csv(conversion_path)

        with st.spinner("Running our magic machine learning model..."):
            raw_predictions = run_forecast(df)
            st.session_state.predictions = apply_unit_conversion(raw_predictions, conversion_df)

        st.session_state.order_results = None
        st.success("Predictions generated successfully!")
    except FileNotFoundError:
        st.error(f"❌ Error: mock_preprocessed_data.csv not found in {Path(__file__).parent}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

# If we have predictions in session state, render the downstream UI
if st.session_state.predictions is not None:
    predictions = st.session_state.predictions

    st.subheader("📊 Forecast Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Items", len(predictions))

    st.subheader("📋 Detailed Predictions")
    st.dataframe(
        predictions[["Product", "Predicted Amount (Next 7 Days)", "Unit"]]
        .style.format({
            "Predicted Amount (Next 7 Days)": "{:.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🧾 Enter Current Inventory")
    input_df = predictions[["Product"]].copy()
    input_df["Current Inventory"] = pd.NA

    try:
        edited_input = st.experimental_data_editor(
            input_df,
            num_rows="fixed",
            use_container_width=True,
            key="inventory_input_editor",
        )
    except Exception:
        edited_input = input_df.copy()
        for i, row in edited_input.iterrows():
            user_val = st.text_input(
                f"{row['Product']} - Current Inventory", value="", key=f"ci_{i}"
            )
            edited_input.at[i, "Current Inventory"] = user_val

    if st.button("Calculate Amount to Order"):
        df_input = edited_input.copy()
        df_input["Current Inventory"] = pd.to_numeric(df_input["Current Inventory"], errors="coerce")

        merged = predictions.merge(df_input, on="Product", how="right")
        merged = merged[merged["Current Inventory"].notna()].copy()

        if not merged.empty:
            merged["Amount to Order"] = np.ceil(
                (merged["Predicted Amount (Next 7 Days)"] - merged["Current Inventory"]).clip(lower=0)
            ).astype(int)

            merged["Predicted Amount (Next 7 Days)"] = merged["Predicted Amount (Next 7 Days)"]
            st.session_state.order_results = merged
        else:
            st.session_state.order_results = pd.DataFrame()
            st.info("No numeric current inventory values provided — enter numbers to calculate orders.")

    if st.session_state.order_results is not None and not st.session_state.order_results.empty:
        display_cols = [
            "Product",
            "Predicted Amount (Next 7 Days)",
            "Current Inventory",
            "Amount to Order",
            "Unit",
        ]

        st.subheader("🧾 Amount to Order")
        st.dataframe(
            st.session_state.order_results[display_cols].style.format({
                "Predicted Amount (Next 7 Days)": "{:.0f}",
                "Current Inventory": "{:.1f}",
                "Amount to Order": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
