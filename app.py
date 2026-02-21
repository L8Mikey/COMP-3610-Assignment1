#816040436-Micah Hosein-app.py- Dashboard
import os
import pandas as pd
import plotly.express as px
import streamlit as st


CLEAN_PATH  = "data/cleaned_taxi_data.parquet"
ZONES_PATH  = "data/raw/taxi_zone_lookup.csv"

PAYMENT_MAP = {
    1: "Credit Card",
    2: "Cash",
    3: "No Charge",
    4: "Dispute",
    0: "Unknown",
    6: "Voided Trip",
}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# Page config 
st.set_page_config(
    page_title="NYC Taxi Dashboard",
    page_icon="🚕",
    layout="wide",
)


# Data loading 

@st.cache_data
def load_trip_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        df[col] = pd.to_datetime(df[col])
    df["payment_label"] = df["payment_type"].map(PAYMENT_MAP).fillna("Other")
    return df


@st.cache_data
def load_zone_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def apply_filters(
    df: pd.DataFrame,
    start_date,
    end_date,
    hour_min: int,
    hour_max: int,
    payment_types: list,
) -> pd.DataFrame:
    
    mask = (
        (df["tpep_pickup_datetime"].dt.date >= start_date)
        & (df["tpep_pickup_datetime"].dt.date <= end_date)
        & (df["pickup_hour"] >= hour_min)
        & (df["pickup_hour"] <= hour_max)
    )
    if payment_types:
        mask &= df["payment_label"].isin(payment_types)
    return df[mask]


# Pre-aggregation helpers 

@st.cache_data
def agg_top_zones(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:

    zone_counts = (
        df.groupby("PULocationID")
        .size()
        .reset_index(name="trip_count")
    )
    merged = zone_counts.merge(
        zones[["LocationID", "Zone", "Borough"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    )
    return merged.nlargest(10, "trip_count")[["Zone", "Borough", "trip_count"]]


@st.cache_data
def agg_fare_by_hour(df: pd.DataFrame) -> pd.DataFrame:
   
    return (
        df.groupby("pickup_hour")["fare_amount"]
        .mean()
        .reset_index()
        .rename(columns={"fare_amount": "avg_fare"})
        .sort_values("pickup_hour")
    )


@st.cache_data
def agg_payment_types(df: pd.DataFrame) -> pd.DataFrame:
   
    return (
        df.groupby("payment_label")
        .size()
        .reset_index(name="trip_count")
        .sort_values("trip_count", ascending=False)
    )


@st.cache_data
def agg_heatmap(df: pd.DataFrame) -> pd.DataFrame:
   
    return (
        df.groupby(["pickup_day_of_week", "pickup_hour"])
        .size()
        .reset_index(name="trip_count")
    )


# Main 

def main():
    if not os.path.exists(CLEAN_PATH):
        st.error(
            f"Cleaned data not found at `{CLEAN_PATH}`. "
            "Please run the notebook first to generate the cleaned parquet file."
        )
        st.stop()

    df_full  = load_trip_data(CLEAN_PATH)
    df_zones = load_zone_data(ZONES_PATH)

    # Title and intro
    st.title("🚕 NYC Yellow Taxi Trip Dashboard")
    st.markdown(
        "Explore January 2024 NYC Yellow Taxi trip patterns, including fare trends, "
        "busy zones, payment methods and time of day behaviour. "
        "Use the sidebar filters to drill into specific segments."
    )

    # Sidebar Filters
    st.sidebar.header("Filters")

    # Date range
    min_date = df_full["tpep_pickup_datetime"].dt.date.min()
    max_date = df_full["tpep_pickup_datetime"].dt.date.max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Guard: date_input returns tuple only when both dates chosen
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range


    # Hour range
    hour_range = st.sidebar.slider(
        "Hour Range (0–23)",
        min_value=0,
        max_value=23,
        value=(0, 23),
    )
    hour_min, hour_max = hour_range


    # Payment type multi-select
    all_payment_types = sorted(df_full["payment_label"].dropna().unique().tolist())

    selected_payments = st.sidebar.multiselect(
        "Payment Types",
        options=all_payment_types,
        default=all_payment_types,
    )


    # Applying filters
    df = apply_filters(df_full, start_date, end_date, hour_min, hour_max, selected_payments)

    if df.empty:
        st.warning("No data matches the current filters. Please adjust the sidebar.")
        st.stop()


    # Key metrics 
    st.subheader("Key Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Trips",        f"{len(df):,}")
    m2.metric("Avg Fare",           f"${df['fare_amount'].mean():.2f}")
    m3.metric("Total Revenue",      f"${df['total_amount'].sum():,.0f}")
    m4.metric("Avg Trip Distance",  f"{df['trip_distance'].mean():.2f} mi")
    m5.metric("Avg Trip Duration",  f"{df['trip_duration_minutes'].mean():.1f} min")

    st.divider()



    # Visualisation 1 — Top 10 Pickup Zones 
    st.subheader("(1) Top 10 Pickup Zones")
    top_zones = agg_top_zones(df, df_zones)

    fig1 = px.bar(
        top_zones.sort_values("trip_count"),
        x="trip_count", y="Zone",
        orientation="h",
        color="trip_count",
        color_continuous_scale="Blues",
        labels={"trip_count": "Number of Trips", "Zone": "Pickup Zone"},
        title="Top 10 Busiest Pickup Zones",
    )

    fig1.update_coloraxes(showscale=False)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "Midtown Center and Upper East Side South lead all zones with over 140,000 trips each, " 
        "reflecting the concentration of business and residential activity in central Manhattan." 
        "JFK Airport's third place ranking highlights how significant airport transfers are " 
        "to overall taxi demand."
    )

    st.divider()


    # Visualisation 2 — Average Fare by Hour
    st.subheader("(2) Average Fare by Hour of Day")
    fare_by_hour = agg_fare_by_hour(df)

    fig2 = px.line(
        fare_by_hour,
        x="pickup_hour", y="avg_fare",
        markers=True,
        labels={"pickup_hour": "Hour of Day (0–23)", "avg_fare": "Avg Fare ($)"},
        title="Average Fare by Hour",
    )

    fig2.update_xaxes(dtick=1)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "The sharp spike at 5 AM is driven by early morning airport runs " 
        "which are longer distance trips than typical city hops. Fares stabilise between " 
        "$17 to 19 during daytime hours before gradually climbing again after 10 PM."
    )

    st.divider()


    # Visualisation 3 — Trip Distance Distribution 
    st.subheader("(3) Trip Distance Distribution")
    sample_size = min(200_000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    fig3 = px.histogram(
        df_sample,
        x="trip_distance",
        nbins=60,
        range_x=[0, 30],
        labels={"trip_distance": "Distance (miles)", "count": "Number of Trips"},
        title="Distribution of Trip Distances (0–30 miles)",
        color_discrete_sequence=["#636EFA"],
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "The vast majority of trips fall under 3 miles, confirming that NYC taxis are primarily " 
        "used for short urban hops within Manhattan. The secondary cluster around 15 to 25 miles " 
        "represents airport trips to JFK and LaGuardia."
    )

    st.divider()


    # Visualisation 4 — Payment Type Breakdown 
    st.subheader("(4) Payment Type Breakdown")
    payment_counts = agg_payment_types(df)

    fig4 = px.pie(
        payment_counts,
        names="payment_label",
        values="trip_count",
        title="Share of Trips by Payment Type",
        hole=0.4,
    )

    fig4.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "Credit card dominates at 80.1%, reflecting the mandatory card reader requirement in NYC " 
        "taxis and the broader shift away from cash payments. Cash at 14.7% remains relevant " 
        "but continues to decline year over year."
    )

    st.divider()


    # Visualisation 5 — Trips by Day of Week × Hour (Heatmap) 
    st.subheader("(5) Trip Volume by Day of Week and Hour")
    heatmap_data = agg_heatmap(df)

    fig5 = px.density_heatmap(
        heatmap_data,
        x="pickup_hour",
        y="pickup_day_of_week",
        z="trip_count",
        category_orders={"pickup_day_of_week": DAY_ORDER},
        color_continuous_scale="YlOrRd",
        labels={
            "pickup_hour": "Hour of Day",
            "pickup_day_of_week": "Day of Week",
            "trip_count": "Trip Count",
        },
        title="Heatmap: Trips by Day of Week and Hour",
    )

    st.plotly_chart(fig5, use_container_width=True)
    st.caption(
        "Tuesday through Thursday afternoons between hours 14 to 17 show the darkest cells, " 
        "pointing to a strong midweek commuter and business travel peak. Weekends spread activity" 
        "more evenly across midday and evening hours rather than peaking sharply in the afternoon."
    )


if __name__ == "__main__":
    main()
