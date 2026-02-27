from collections import defaultdict
from pathlib import Path
import sqlite3

import streamlit as st
import altair as alt
import pandas as pd
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


# -----------------------------------------------------------------------------
# Draw the actual page, starting with the inventory table.

# Set the title that appears at the top of the page.
"""
# Compass Coffee Reorder Dashboard

**Welcome to Compass Coffee's reorder dashboard for store managers!**
Using recent sales data you upload, this page provides inventory stocking recommendations using Machine Learning. These are only suggestions, and the developers of this page are not responsible for any errors. For internal use only.
"""

# Store location selector
st.subheader("Select Store Location")
store_location = st.selectbox(
    "Choose your store:",
    [
        "Store 1",
        "Store 2",
        "Store 3",
        "Store 4",
        "Store 5",
    ],
)

# CSV file upload for demand prediction
st.subheader("Upload Recent Sales Data")
uploaded_file = st.file_uploader(
    "Upload a CSV file for demand prediction",
    type="csv",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

# Prediction section
st.subheader("Generate Demand Predictions")
st.write("Click the button below to generate demand predictions for the next 7 days.")

# Button to trigger prediction
if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
    try:
        # Load mock preprocessed data
        data_path = Path(__file__).parent / "mock_preprocessed_data.csv"
        df = pd.read_csv(data_path)
        
        # Run the forecast model
        with st.spinner("Running our magic machine learning model..."):
            predictions = run_forecast(df)
        
        st.success("Predictions generated successfully!")
        
        # Display metrics summary
        st.subheader("📊 Forecast Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Items", len(predictions))
        with col2:
            st.metric("Top Item Demand", f"{predictions['Predicted Amount (Next 7 Days)'].iloc[0]:.1f}")

        # Display full predictions table (read-only)
        st.subheader("📋 Detailed Predictions")
        st.dataframe(
            predictions.style.format({"Predicted Amount (Next 7 Days)": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

        # Inventory input table: Product + Current Inventory (editable)
        st.subheader("🧾 Enter Current Inventory")
        input_df = predictions[["Product"]].copy()
        input_df["Current Inventory"] = pd.NA

        try:
            edited_input = st.experimental_data_editor(
                input_df,
                num_rows="fixed",
                use_container_width=True,
            )
        except Exception:
            edited_input = input_df.copy()
            for i, row in edited_input.iterrows():
                user_val = st.text_input(f"{row['Product']} - Current Inventory", value="", key=f"ci_{i}")
                edited_input.at[i, "Current Inventory"] = user_val

        # Calculate order amounts when requested
        if st.button("Calculate Amount to Order"):
            df_input = edited_input.copy()
            df_input["Current Inventory"] = pd.to_numeric(df_input["Current Inventory"], errors="coerce")

            # Merge with predictions to get predicted amounts
            merged = predictions.merge(
                df_input, on="Product", how="right", suffixes=(None, None)
            )

            # Keep only rows where user provided a numeric current inventory
            merged = merged[merged["Current Inventory"].notna()].copy()

            if not merged.empty:
                merged["Amount to Order"] = (
                    merged["Predicted Amount (Next 7 Days)"] - merged["Current Inventory"]
                ).clip(lower=0)

                display_cols = [
                    "Product",
                    "Predicted Amount (Next 7 Days)",
                    "Current Inventory",
                    "Amount to Order",
                ]

                st.subheader("🧾 Amount to Order")
                st.dataframe(
                    merged[display_cols].style.format({
                        "Predicted Amount (Next 7 Days)": "{:.1f}",
                        "Current Inventory": "{:.1f}",
                        "Amount to Order": "{:.1f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No numeric current inventory values provided — enter numbers to calculate orders.")
        
    except FileNotFoundError:
        st.error(f"❌ Error: mock_preprocessed_data.csv not found in {Path(__file__).parent}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
