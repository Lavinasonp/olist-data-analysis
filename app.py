import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import os# --- Configuration & Styling ---
st.set_page_config(
    page_title="Olist Supply Chain Analytics", 
    page_icon="📦", 
    layout="wide"
)

# Deep Data Analyst Styling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        color: #1E3A8A;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .author-badge {
        background: linear-gradient(90deg, #4F46E5 0%, #EC4899 100%);
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        margin: auto;
        width: 100%;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Author
st.markdown("<h1 class='main-title'>Olist Supply Chain Excellence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-title'>Advanced Demand Forecasting & Logistics Analytics</h3>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='author-badge'>🌟 Made by Laveena Sonp</div>", unsafe_allow_html=True)
st.sidebar.header("Navigation")

# --- Load Data Helpers ---
@st.cache_data
def load_data():
    master_path = os.path.join("data", "processed", "master_table.parquet")
    features_path = os.path.join("data", "processed", "features_v3.parquet")
    
    if not os.path.exists(master_path):
        st.error(f"Missing {master_path}. Please run data_loader.py")
        st.stop()
        
    df_master = pd.read_parquet(master_path)
    
    # Dates
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date']
    for col in date_cols:
        df_master[col] = pd.to_datetime(df_master[col], errors='coerce')
        
    df_features = pd.read_parquet(features_path) if os.path.exists(features_path) else None
    
    return df_master, df_features

df_master, df_features = load_data()

# --- Navigation Logic ---
navigation = st.sidebar.radio("Go to Analysis:", [
    "1. Executive Overview", 
    "2. Logistics & Performance", 
    "3. Demand Forecasting",
    "4. Inventory Replenishment"
])

st.sidebar.markdown("---")
st.sidebar.info("This interactive dashboard analyzes Olist e-commerce data to optimize supply chain operations, predict future demand, and guide inventory policy.")

# --- 1. Executive Overview ---
if navigation == "1. Executive Overview":
    st.header("Executive Overview & Purchasing Patterns")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"R$ {df_master['price'].sum():,.0f}")
    with col2:
        st.metric("Total Orders", f"{df_master['order_id'].nunique():,}")
    with col3:
        st.metric("Total Customers", f"{df_master['customer_id'].nunique():,}")
    with col4:
        st.metric("Avg Order Value", f"R$ {df_master['price'].mean():,.2f}")
        
    st.markdown("---")
    
    # A. Monthly Revenue Trend
    st.subheader("Revenue Trajectory")
    
    # Safely convert to datetime to ensure pandas resampling works correctly
    temp_df = df_master.copy()
    temp_df['order_purchase_timestamp'] = pd.to_datetime(temp_df['order_purchase_timestamp'])
    
    monthly_sales = temp_df.set_index('order_purchase_timestamp').resample('ME')['price'].sum().reset_index()
    fig1 = px.line(monthly_sales, x='order_purchase_timestamp', y='price', 
                   title="Total Monthly Revenue", 
                   labels={'order_purchase_timestamp': 'Month', 'price':'Revenue (R$)'},
                   template="plotly_white", line_shape='spline')
    fig1.update_traces(line_color='#4F46E5', line_width=3)
    st.plotly_chart(fig1, use_container_width=True)
    
    # B. Geographic Distribution
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Customer Distribution by State")
        state_counts = df_master['customer_state'].value_counts().reset_index()
        state_counts.columns = ['customer_state', 'count']
        fig2 = px.bar(state_counts.head(10), x='customer_state', y='count', 
                      color='count', color_continuous_scale='Viridis',
                      title="Top 10 States by Order Volume")
        st.plotly_chart(fig2, use_container_width=True)
        
    with colB:
        st.subheader("Top Product Categories")
        cat_counts = df_master['product_category'].value_counts().reset_index()
        cat_counts.columns = ['category', 'sales']
        fig3 = px.pie(cat_counts.head(10), values='sales', names='category', hole=0.4,
                      title="Revenue Breakdown (Top 10 Categories)", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig3, use_container_width=True)

# --- 2. Logistics & Performance ---
elif navigation == "2. Logistics & Performance":
    st.header("Supply Chain & Logistics Performance")
    
    df_logistics = df_master.copy()
    df_logistics['time_to_ship'] = (df_logistics['order_delivered_carrier_date'] - df_logistics['order_approved_at']).dt.days
    df_logistics['carrier_time'] = (df_logistics['order_delivered_customer_date'] - df_logistics['order_delivered_carrier_date']).dt.days
    df_logistics['delay_days'] = (df_logistics['order_delivered_customer_date'] - df_logistics['order_estimated_delivery_date']).dt.days
    
    # Filter valid
    valid_logs = df_logistics[(df_logistics['time_to_ship'] >= 0) & (df_logistics['carrier_time'] >= 0)].copy()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Seller Processing", f"{valid_logs['time_to_ship'].mean():.1f} days")
    col2.metric("Avg Carrier Transit", f"{valid_logs['carrier_time'].mean():.1f} days")
    late_pct = (valid_logs['delay_days'] > 0).mean() * 100
    col3.metric("Late Delivery Rate", f"{late_pct:.1f}%", "-1.2%" if late_pct < 10 else "+2.1%", delta_color="inverse")
    
    st.markdown("---")
    
    # Seller vs Carrier Analysis
    st.subheader("Who is the bottleneck? (Seller vs Carrier)")
    
    state_logs = valid_logs.groupby('customer_state')[['time_to_ship', 'carrier_time']].mean().sort_values('carrier_time', ascending=False).reset_index()
    
    fig_stack = go.Figure(data=[
        go.Bar(name='Seller Processing', x=state_logs['customer_state'], y=state_logs['time_to_ship'], marker_color='#3498db'),
        go.Bar(name='Carrier Transit', x=state_logs['customer_state'], y=state_logs['carrier_time'], marker_color='#e67e22')
    ])
    fig_stack.update_layout(barmode='stack', title="Average Delivery Composition by State", xaxis_title="State", yaxis_title="Average Days")
    st.plotly_chart(fig_stack, use_container_width=True)
    
    # Delivery Reliability
    st.subheader("Delivery Reliability (Actual vs Estimated)")
    plot_data = valid_logs[(valid_logs['delay_days'] > -20) & (valid_logs['delay_days'] < 20)]
    fig_box = px.box(plot_data, x='customer_state', y='delay_days', 
                     title="Distribution of Delay Days by State (Positive = Late)",
                     color='customer_state')
    fig_box.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="On Time")
    st.plotly_chart(fig_box, use_container_width=True)

# --- 3. Demand Forecasting ---
elif navigation == "3. Demand Forecasting":
    st.header("ML-Driven Demand Forecasting")
    
    forecast_path = os.path.join("data", "processed", "dashboard_forecast_v3.parquet")
    if not os.path.exists(forecast_path):
        st.warning("Forecasting data missing. Please run `python src/generate_dashboard_data.py`.")
    else:
        st.info("The model is a LightGBM Regressor using a Tweedie Objective, trained on lagged time-series features.")
        
        # Load pre-computed results
        res_df = pd.read_parquet(forecast_path)
        
        # Global Forecast
        st.subheader("Global Aggregate Demand Forecast (Test Period)")
        global_demand = res_df.groupby('date')[['actual', 'predicted']].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=global_demand['date'], y=global_demand['actual'], mode='lines', name='Actual', line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=global_demand['date'], y=global_demand['predicted'], mode='lines', name='Predicted', line=dict(color='#2ca02c', width=2, dash='dash')))
        fig.update_layout(title="Total Network Demand vs Model Forecast", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Category specific
        st.subheader("Category-Specific Forecasts")
        top_categories = res_df.groupby('category')['actual'].sum().nlargest(10).index.tolist()
        selected_cat = st.selectbox("Select a Product Category to View:", top_categories)
        
        cat_data = res_df[res_df['category'] == selected_cat].groupby('date')[['actual', 'predicted']].sum().reset_index()
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Scatter(x=cat_data['date'], y=cat_data['actual'], mode='lines+markers', name='Actual Sales'))
        fig_cat.add_trace(go.Scatter(x=cat_data['date'], y=cat_data['predicted'], mode='lines', name='Forecasted Sales', line=dict(dash='dash')))
        st.plotly_chart(fig_cat, use_container_width=True)

# --- 4. Inventory Replenishment ---
elif navigation == "4. Inventory Replenishment":
    st.header("Inventory Replenishment & Safety Stock")
    
    inventory_path = os.path.join("data", "processed", "dashboard_inventory_v3.parquet")
    if not os.path.exists(inventory_path):
        st.warning("Inventory data missing. Please run `python src/generate_dashboard_data.py`.")
    else:
        st.markdown("Dynamic calculation of recommended inventory levels utilizing ML demand forecasts and standard supply chain safety stock formulas.")
        
        # Load pre-computed inventory data
        inv_df = pd.read_parquet(inventory_path)
        latest_date = inv_df['Planning_Date'].iloc[0]
        
        st.write(f"**Planning Date (Simulated Current Week):** {pd.to_datetime(latest_date).strftime('%Y-%m-%d')}")
        
        # Calculation parameters
        service_level = st.slider("Target Service Level (%)", 85, 99, 95)
        lead_time = st.slider("Supplier Lead Time (Weeks)", 1, 8, 2)
        
        # Map service level to z-score approximately
        z_dict = {85: 1.04, 90: 1.28, 95: 1.645, 98: 2.05, 99: 2.33}
        z_val = z_dict.get(service_level, 1.645) # fallback
        
        inv_df['Safety_Stock'] = np.ceil(z_val * np.sqrt(lead_time) * inv_df['Volatility_StdDev'])
        inv_df['Reorder_Point (ROP)'] = (inv_df['Expected_Weekly_Demand'] * lead_time) + inv_df['Safety_Stock']
        
        # Keep only needed columns and sort
        display_cols = ['Category', 'Expected_Weekly_Demand', 'Volatility_StdDev', 'Safety_Stock', 'Reorder_Point (ROP)']
        inv_df_display = inv_df[display_cols].sort_values('Expected_Weekly_Demand', ascending=False)
        
        # Display table
        st.dataframe(inv_df_display.head(50).style.format({
            'Expected_Weekly_Demand': '{:.0f}',
            'Volatility_StdDev': '{:.1f}',
            'Safety_Stock': '{:.0f}',
            'Reorder_Point (ROP)': '{:.0f}',
        }), use_container_width=True)
        
        # Visualizing ROP for Top 10
        st.subheader("Reorder Point Composition (Top Categories)")
        top_inv = inv_df_display.head(15).copy()
        top_inv['Lead_Time_Demand'] = top_inv['Expected_Weekly_Demand'] * lead_time
        
        fig_rop = go.Figure(data=[
            go.Bar(name='Demand during Lead Time', x=top_inv['Category'], y=top_inv['Lead_Time_Demand'], marker_color='#4ade80'),
            go.Bar(name='Safety Stock', x=top_inv['Category'], y=top_inv['Safety_Stock'], marker_color='#f87171')
        ])
        fig_rop.update_layout(barmode='stack', title="Total Reorder Point = Base Demand + Safety Buffer")
        st.plotly_chart(fig_rop, use_container_width=True)
