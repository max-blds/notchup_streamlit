# Open in Notepad or text editor and paste this:

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from gspread_dataframe import get_as_dataframe
from datetime import datetime, timedelta
import json
import os

# Your data loading function
def load_and_preprocess_data():
    # Load data
    # Load credentials from Streamlit secrets
    credentials_json = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    gc = gspread.service_account_from_dict(credentials_json)
    sh = gc.open("Notchup Cohort Analysis")
    worksheet = sh.worksheet("Raw Data")
    df = get_as_dataframe(worksheet)
    
    # Clean up - remove empty rows/columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Convert amount columns to numeric
    df['Principle'] = pd.to_numeric(df['Principle'], errors='coerce')
    df['Total Paid'] = pd.to_numeric(df['Total Paid'], errors='coerce')
    
    # Calculate Profit
    df['Profit'] = df['Total Paid'] - df['Principle']
    
    # Convert dates
    date_columns = ['Created', 'Loan Date', 'Payment Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df

# Second cell - analysis functions
def analyze_loan_data(df, min_users=12):
    # Convert dates to datetime
    df['Created'] = pd.to_datetime(df['Created'])
    df['Loan Date'] = pd.to_datetime(df['Loan Date'])
    df['Payment Date'] = pd.to_datetime(df['Payment Date'])
    
    # Create monthly cohort
    df['Cohort'] = df['Created'].dt.strftime('%Y-%m')
    
    # Filter out pending loans and rows with missing cohorts
    active_df = df[
        (df['Status'] != 'Pending') & 
        (df['Cohort'].notna())
    ].copy()
    
    # Calculate users per cohort and filter
    users_per_cohort = active_df.groupby('Cohort')['user_recid'].nunique()
    small_cohorts = users_per_cohort[users_per_cohort < min_users].index
    if len(small_cohorts) > 0:
        print(f"\nDropping {len(small_cohorts)} small cohorts:")
        for cohort in sorted(small_cohorts):
            print(f"Cohort {cohort}: {users_per_cohort[cohort]} users")
    
    active_df = active_df[~active_df['Cohort'].isin(small_cohorts)]
    
    # Create loan sequence number for each user
    active_df = active_df.sort_values(['user_recid', 'Loan Date'])
    active_df['loan_num'] = active_df.groupby('user_recid').cumcount() + 1
    
    # Create binary default column
    active_df['is_default'] = active_df['Status'].str.contains('Default', case=False).astype(int)
    
    # Calculate overall cohort default rates
    cohort_defaults = active_df.groupby('Cohort').agg({
        'is_default': ['count', 'mean'],
        'user_recid': 'nunique',
        'Principle': 'mean'
    }).round(4)
    
    cohort_defaults.columns = ['Total_Loans', 'Default_Rate', 'Total_Users', 'Avg_Principal']
    cohort_defaults = cohort_defaults.reset_index()
    
    # Calculate metrics by loan sequence number
    sequence_metrics = active_df.groupby('loan_num').agg({
        'is_default': ['count', 'mean'],
        'Principle': 'mean'
    }).reset_index()
    
    sequence_metrics.columns = ['loan_num', 'loan_count', 'default_rate', 'avg_principal']
    
    return active_df, cohort_defaults, sequence_metrics
    pass

def analyze_ltv_and_retention(df, min_users=12):
    # Convert dates to datetime
    df['Created'] = pd.to_datetime(df['Created'])
    df['Loan Date'] = pd.to_datetime(df['Loan Date'])
    df['Payment Date'] = pd.to_datetime(df['Payment Date'])
    
    # Create monthly cohort and handle NaN values
    df['Cohort'] = df['Created'].dt.strftime('%Y-%m')
    
    # Filter out pending loans and rows with missing cohorts
    active_df = df[
        (df['Status'] != 'Pending') & 
        (df['Cohort'].notna())
    ].copy()
    
    # Calculate users per cohort and filter
    users_per_cohort = active_df.groupby('Cohort')['user_recid'].nunique()
    small_cohorts = users_per_cohort[users_per_cohort < min_users].index
    if len(small_cohorts) > 0:
        print(f"\nDropping {len(small_cohorts)} small cohorts:")
        for cohort in sorted(small_cohorts):
            print(f"Cohort {cohort}: {users_per_cohort[cohort]} users")
    
    active_df = active_df[~active_df['Cohort'].isin(small_cohorts)]
    
    # Calculate week number for each loan relative to cohort start
    def get_week_number(row):
        cohort_start = active_df[active_df['Cohort'] == row['Cohort']]['Created'].min()
        weeks = (row['Loan Date'] - cohort_start).days // 7
        return max(0, weeks)
    
    active_df['Week_Number'] = active_df.apply(get_week_number, axis=1)
    
    # Calculate metrics by cohort and week
    ltv_by_week = []
    retention_by_week = []
    retention_metrics = []
    
    latest_date = active_df['Loan Date'].max()
    
    # Make sure to convert profit column to numeric
    active_df['Profit'] = pd.to_numeric(active_df['Profit'], errors='coerce')
    
    print("\nProcessing remaining cohorts:")
    for cohort in sorted(active_df['Cohort'].unique()):
        cohort_data = active_df[active_df['Cohort'] == cohort]
        cohort_users = len(cohort_data['user_recid'].unique())
        print(f"Cohort {cohort}: {cohort_users} users")
            
        # Calculate cumulative profit by week
        weekly_profits = cohort_data.groupby('Week_Number')['Profit'].sum()
        cumulative_profits = weekly_profits.cumsum()
        
        # Fill in missing weeks with previous cumulative value
        all_weeks = pd.Series(index=range(max(cumulative_profits.index) + 1))
        cumulative_profits = cumulative_profits.reindex(all_weeks.index).ffill()
        
        for week in range(max(cumulative_profits.index) + 1):
            week_date = cohort_data['Created'].min() + timedelta(weeks=week)
            thirty_days_before = week_date - timedelta(days=30)
            
            # Calculate loans and users up to this week
            loans_until_week = active_df[
                (active_df['Cohort'] == cohort) & 
                (active_df['Week_Number'] <= week)
            ]
            
            # Users who haven't defaulted by this week
            user_status = loans_until_week.groupby('user_recid')['Status'].apply(
                lambda x: ~x.str.contains('Default', case=False).any()
            )
            good_users = len(user_status[user_status])
            
            # Active users (had a loan in last 30 days of this week)
            recent_loans = active_df[
                (active_df['Cohort'] == cohort) & 
                (active_df['Loan Date'] <= week_date) &
                (active_df['Loan Date'] > thirty_days_before)
            ]
            active_users = len(recent_loans['user_recid'].unique())
            
            # Active and good users
            active_good_users = len(
                recent_loans[
                    recent_loans['user_recid'].isin(user_status[user_status].index)
                ]['user_recid'].unique()
            )
            
            # LTV metrics
            cum_profit = cumulative_profits.get(week, cumulative_profits.iloc[-1])
            ltv_by_week.append({
                'Cohort': cohort,
                'Week_Number': int(week),
                'Cumulative_Profit': float(cum_profit),
                'Users': cohort_users,
                'Cumulative_LTV': float(cum_profit) / cohort_users if cohort_users > 0 else 0
            })
            
            # Retention metrics
            retention_by_week.append({
                'Cohort': cohort,
                'Week_Number': week,
                'Good_Users': good_users,
                'Good_Users_Pct': (good_users / cohort_users * 100) if cohort_users > 0 else 0,
                'Active_Users': active_users,
                'Active_Users_Pct': (active_users / cohort_users * 100) if cohort_users > 0 else 0,
                'Active_Good_Users': active_good_users,
                'Active_Good_Users_Pct': (active_good_users / cohort_users * 100) if cohort_users > 0 else 0
            })
    
    return (pd.DataFrame(ltv_by_week), 
            pd.DataFrame(retention_by_week))
    pass

# Third cell - create plotly versions of your visualizations
def plot_combined_metrics_plotly(cohort_defaults, sequence_metrics):
    # Create 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Default Rate by Cohort', 
                       'Number of Loans by Sequence',
                       'Default Rate by Loan Sequence', 
                       'Average Principal by Sequence')
    )
    
    # Plot 1: Cohort Default Rates
    fig.add_trace(
        go.Bar(x=cohort_defaults['Cohort'],
               y=cohort_defaults['Default_Rate'] * 100,
               text=cohort_defaults['Default_Rate'].mul(100).round(1).astype(str) + '%',
               textposition='auto',
               name='Default Rate'),
        row=1, col=1
    )
    
    # Plot 2: Loan Counts by Sequence
    fig.add_trace(
        go.Bar(x=sequence_metrics['loan_num'],
               y=sequence_metrics['loan_count'],
               text=sequence_metrics['loan_count'],
               textposition='auto',
               name='Loan Count',
               marker_color='green'),
        row=1, col=2
    )
    
    # Plot 3: Default Rates by Sequence
    fig.add_trace(
        go.Scatter(x=sequence_metrics['loan_num'],
                  y=sequence_metrics['default_rate'] * 100,
                  mode='lines+markers+text',
                  text=sequence_metrics['default_rate'].mul(100).round(1).astype(str) + '%',
                  textposition='top center',
                  name='Default Rate',
                  line=dict(color='red')),
        row=2, col=1
    )
    
    # Plot 4: Average Principal by Sequence
    fig.add_trace(
        go.Bar(x=sequence_metrics['loan_num'],
               y=sequence_metrics['avg_principal'],
               text='$' + sequence_metrics['avg_principal'].round(0).astype(str),
               textposition='auto',
               name='Avg Principal',
               marker_color='blue'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=800, 
                     showlegend=False,
                     title_text="Loan Metrics Analysis")
    
    return fig

def plot_cohort_metrics_plotly(ltv_df, retention_df):
    # Create three subplots vertically
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=('Cumulative LTV by Monthly Cohort',
                                      'Never Defaulted Users by Cohort',
                                      '30-Day Active Users by Cohort'))
    
    # Plot 1: Cumulative LTV
    for cohort in sorted(ltv_df['Cohort'].unique()):
        cohort_data = ltv_df[ltv_df['Cohort'] == cohort]
        fig.add_trace(
            go.Scatter(x=cohort_data['Week_Number'],
                      y=cohort_data['Cumulative_LTV'],
                      mode='lines+markers',
                      name=f'Cohort {cohort}'),
            row=1, col=1
        )
    
    # Plot 2: Good Users Retention
    for cohort in sorted(retention_df['Cohort'].unique()):
        cohort_data = retention_df[retention_df['Cohort'] == cohort]
        fig.add_trace(
            go.Scatter(x=cohort_data['Week_Number'],
                      y=cohort_data['Good_Users_Pct'],
                      mode='lines+markers',
                      name=f'Cohort {cohort}'),
            row=2, col=1
        )
    
    # Plot 3: Active Users Retention
    for cohort in sorted(retention_df['Cohort'].unique()):
        cohort_data = retention_df[retention_df['Cohort'] == cohort]
        fig.add_trace(
            go.Scatter(x=cohort_data['Week_Number'],
                      y=cohort_data['Active_Users_Pct'],
                      mode='lines+markers',
                      name=f'Cohort {cohort}'),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(height=1200,
                     showlegend=True,
                     legend=dict(
                         yanchor="top",
                         y=0.99,
                         xanchor="left",
                         x=1.05
                     ))
    
    # Update y-axes ranges
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    return fig

# Streamlit app code
st.title("Notchup Loan Analytics")

try:
    # Load data
    df = load_and_preprocess_data()
    
    with st.spinner('Analyzing loan data...'):
        active_df, cohort_defaults, sequence_metrics = analyze_loan_data(df, min_users=12)
        ltv_df, retention_df = analyze_ltv_and_retention(df, min_users=12)

    # Create and display plots using st.plotly_chart instead of fig.show()
    st.subheader("Loan Metrics Analysis")
    fig1 = plot_combined_metrics_plotly(cohort_defaults, sequence_metrics)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Cohort Metrics Analysis")
    fig2 = plot_cohort_metrics_plotly(ltv_df, retention_df)
    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

