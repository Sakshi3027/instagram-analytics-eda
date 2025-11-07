"""
Instagram Analytics Interactive Dashboard
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Instagram Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Instagram Content Performance Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/Instagram-data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate engagement metrics
    df['Engagement_Rate'] = ((df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']) / df['Impressions']) * 100
    df['Like_Rate'] = (df['Likes'] / df['Impressions']) * 100
    df['Save_Rate'] = (df['Saves'] / df['Impressions']) * 100
    df['Comment_Rate'] = (df['Comments'] / df['Impressions']) * 100
    df['Share_Rate'] = (df['Shares'] / df['Impressions']) * 100
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month_name()
    
    return df

# Load the data
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Filter data based on date range
if len(date_range) == 2:
    mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
    filtered_df = df[mask]
else:
    filtered_df = df

# Engagement filter
min_engagement = st.sidebar.slider(
    "Minimum Engagement Rate (%)",
    min_value=0.0,
    max_value=float(df['Engagement_Rate'].max()),
    value=0.0,
    step=0.5
)

filtered_df = filtered_df[filtered_df['Engagement_Rate'] >= min_engagement]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing {len(filtered_df)} of {len(df)} posts**")

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Posts",
        value=f"{len(filtered_df):,}",
        delta=None
    )

with col2:
    st.metric(
        label="Total Impressions",
        value=f"{filtered_df['Impressions'].sum():,.0f}",
        delta=f"{filtered_df['Impressions'].mean():,.0f} avg"
    )

with col3:
    st.metric(
        label="Total Likes",
        value=f"{filtered_df['Likes'].sum():,.0f}",
        delta=f"{filtered_df['Likes'].mean():.0f} avg"
    )

with col4:
    st.metric(
        label="Avg Engagement",
        value=f"{filtered_df['Engagement_Rate'].mean():.2f}%",
        delta=None
    )

st.markdown("---")

# Row 1: Time series and engagement distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("Impressions Over Time")
    fig_time = px.line(
        filtered_df,
        x='Date',
        y='Impressions',
        title='',
        markers=True
    )
    fig_time.update_layout(
        xaxis_title="Date",
        yaxis_title="Impressions",
        hovermode='x unified'
    )
    st.plotly_chart(fig_time, use_container_width=True)

with col2:
    st.subheader("Engagement Rate Distribution")
    fig_eng = px.histogram(
        filtered_df,
        x='Engagement_Rate',
        nbins=30,
        title=''
    )
    fig_eng.update_layout(
        xaxis_title="Engagement Rate (%)",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig_eng, use_container_width=True)

st.markdown("---")

# Row 2: Traffic sources and day of week
col1, col2 = st.columns(2)

with col1:
    st.subheader("Traffic Sources")
    source_data = pd.DataFrame({
        'Source': ['Home', 'Hashtags', 'Explore', 'Other'],
        'Impressions': [
            filtered_df['From Home'].sum(),
            filtered_df['From Hashtags'].sum(),
            filtered_df['From Explore'].sum(),
            filtered_df['From Other'].sum()
        ]
    })
    
    fig_sources = px.pie(
        source_data,
        values='Impressions',
        names='Source',
        title='',
        hole=0.4
    )
    st.plotly_chart(fig_sources, use_container_width=True)

with col2:
    st.subheader("Performance by Day of Week")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_data = filtered_df.groupby('Day_of_Week')['Impressions'].mean().reindex(
        [d for d in day_order if d in filtered_df['Day_of_Week'].unique()]
    ).reset_index()
    
    fig_dow = px.bar(
        dow_data,
        x='Day_of_Week',
        y='Impressions',
        title=''
    )
    fig_dow.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Average Impressions"
    )
    st.plotly_chart(fig_dow, use_container_width=True)

st.markdown("---")

# Row 3: Scatter plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Likes vs Impressions")
    fig_scatter1 = px.scatter(
        filtered_df,
        x='Impressions',
        y='Likes',
        size='Engagement_Rate',
        color='Engagement_Rate',
        hover_data=['Date'],
        title=''
    )
    st.plotly_chart(fig_scatter1, use_container_width=True)

with col2:
    st.subheader("Profile Visits vs Follows")
    fig_scatter2 = px.scatter(
        filtered_df,
        x='Profile Visits',
        y='Follows',
        size='Impressions',
        color='Engagement_Rate',
        hover_data=['Date'],
        title=''
    )
    st.plotly_chart(fig_scatter2, use_container_width=True)

st.markdown("---")

# Engagement metrics comparison
st.subheader("Average Engagement Metrics")
metrics_data = pd.DataFrame({
    'Metric': ['Likes', 'Comments', 'Shares', 'Saves'],
    'Average': [
        filtered_df['Likes'].mean(),
        filtered_df['Comments'].mean(),
        filtered_df['Shares'].mean(),
        filtered_df['Saves'].mean()
    ]
})

fig_metrics = px.bar(
    metrics_data,
    x='Metric',
    y='Average',
    title='',
    color='Metric'
)
st.plotly_chart(fig_metrics, use_container_width=True)

st.markdown("---")

# Top performing posts
st.subheader("🏆 Top 10 Performing Posts")
top_posts = filtered_df.nlargest(10, 'Engagement_Rate')[
    ['Date', 'Impressions', 'Likes', 'Comments', 'Shares', 'Saves', 'Engagement_Rate']
].copy()
top_posts['Date'] = top_posts['Date'].dt.strftime('%Y-%m-%d')
top_posts['Engagement_Rate'] = top_posts['Engagement_Rate'].round(2)

st.dataframe(
    top_posts,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Correlation heatmap
st.subheader("Correlation Matrix")
corr_cols = ['Impressions', 'Likes', 'Comments', 'Shares', 'Saves', 'Profile Visits', 'Follows']
corr_matrix = filtered_df[corr_cols].corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto='.2f',
    aspect='auto',
    color_continuous_scale='RdBu_r'
)
fig_corr.update_layout(
    title='',
    width=800,
    height=600
)
st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Instagram Analytics Dashboard | Built by Sakshi | Data Science Project</p>
    </div>
""", unsafe_allow_html=True)