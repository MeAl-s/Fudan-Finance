import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key
# Initialize LangChain model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Generate sample P2P lending data with embedded risks
def generate_sample_data():
    data = {
        "platform_id": ["P2P-001", "P2P-007", "P2P-003", "P2P-007", "P2P-005", "P2P-002"],
        "loan_id": ["L-1023", "L-5512", "L-9917", "L-3381", "L-4476", "L-8890"],
        "borrower_id": ["B-88921", "B-00238", "B-77432", "B-00991", "B-556123", "B-00238"],
        "amount": [50000, 200000, 80000, 350000, 120000, 150000],
        "interest_rate": [18.5, 24.9, 15.2, 32.5, 12.8, 28.5],
        "loan_term": [12, 24, 36, 6, 24, 3],
        "borrower_income": [120000, 65000, 185000, 42000, 95000, 68000],
        "credit_score": [715, 682, 781, 605, 698, 621],
        "loan_purpose": ["Business expansion", "Medical emergency", "Home renovation", 
                         "Debt consolidation", "Education loan", "Vacation"],
        "collateral_value": [75000, 0, 120000, 0, 180000, 0],
        "platform_license": ["LIC-A385", "NO-LICENSE", "LIC-C992", "NO-LICENSE", "SUSPENDED", "LIC-B441"],
        "kyc_status": ["Verified", "Expired", "Verified", "Pending", "Verified", "Verified"],
        "transaction_date": ["2023-05-12", "2023-06-18", "2023-07-05", "2023-08-22", "2023-09-14", "2023-10-05"],
        "repayment_status": ["Delayed", "Defaulted", "Current", "Delayed", "Current", "Current"],
        "platform_capital_ratio": [8.2, 3.1, 12.5, 2.8, 6.7, 9.2],
        "related_party_flag": ["No", "Yes", "No", "Yes", "No", "No"]
    }
    return pd.DataFrame(data)

# LLM analysis function
def analyze_with_llm(df):
    prompt = f"""
    Analyze this P2P lending data for regulatory risks. Focus on:
    1. Unlicensed platforms (NO-LICENSE/SUSPENDED in platform_license)
    2. Capital adequacy violations (platform_capital_ratio < 8%)
    3. Predatory lending (interest_rate > 24% AND credit_score < 650)
    4. Suspicious related-party transactions (related_party_flag = 'Yes')
    5. KYC/AML compliance gaps (kyc_status != 'Verified')
    6. Loan churning (multiple loans to same borrower_id)
    
    Data Sample:
    {df.head(3).to_csv(index=False)}
    
    Provide:
    - Risk category for each flagged loan
    - Short explanation of the violation
    - Recommended regulatory action
    - Confidence level (High/Medium/Low)
    
    Format as markdown table with columns: Loan ID, Risk Category, Explanation, Action, Confidence
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# Risk highlighting for DataFrame
def highlight_risks(row):
    styles = [''] * len(row)
    
    # Unlicensed platform
    if row['platform_license'] in ['NO-LICENSE', 'SUSPENDED']:
        styles[0] = 'background-color: #FFCCCB'  # Light red
        
    # Capital adequacy
    if row['platform_capital_ratio'] < 8:
        styles[14] = 'background-color: #FFD580'  # Light orange
        
    # Predatory lending
    if row['interest_rate'] > 24 and row['credit_score'] < 650:
        styles[4] = 'background-color: #FF6666'   # Stronger red
        styles[7] = 'background-color: #FF6666'
        
    # Related party
    if row['related_party_flag'] == 'Yes':
        styles[15] = 'background-color: #90EE90'  # Light green
        
    # KYC issues
    if row['kyc_status'] != 'Verified':
        styles[11] = 'background-color: #ADD8E6'  # Light blue
        
    return styles

# Dashboard visualization using Streamlit native charts
def create_dashboard(df):
    tab1, tab2, tab3 = st.tabs(["License Compliance", "Risk Exposure", "Borrower Analysis"])
    
    with tab1:
        st.subheader("Platform License Status")
        license_counts = df['platform_license'].apply(
            lambda x: "Valid" if x.startswith('LIC') else "Invalid"
        ).value_counts()
        st.bar_chart(license_counts)
        
    with tab2:
        st.subheader("Platform Risk Exposure")
        # Calculate risk scores
        conditions = [
            (df['interest_rate'] > 24) & (df['credit_score'] < 650),
            df['platform_license'].isin(['NO-LICENSE', 'SUSPENDED']),
            df['platform_capital_ratio'] < 8
        ]
        choices = [3, 2, 1]
        df['risk_score'] = np.select(conditions, choices, default=0)
        
        # Aggregate risk by platform
        platform_risk = df.groupby('platform_id')['risk_score'].max().sort_values()
        st.bar_chart(platform_risk)
        
    with tab3:
        st.subheader("Credit Score vs Interest Rate")
        # Use Streamlit's native scatter chart
        chart_data = df[['credit_score', 'interest_rate', 'amount', 'repayment_status']].copy()
        chart_data['size'] = chart_data['amount'] / 10000  # Scale for bubble size
        st.scatter_chart(
            chart_data,
            x='credit_score',
            y='interest_rate',
            size='size',
            color='repayment_status'
        )

# Streamlit app
def main():
    st.set_page_config(
        page_title="P2P Lending RegTech Monitor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("小龙虾 P2P")
    st.caption("Leveraging GPT-4o to detect regulatory risks in peer-to-peer lending platforms")
    
    # Data upload section
    uploaded_file = st.file_uploader("Upload P2P Loan Data (CSV)", type="csv")
    df = generate_sample_data() if uploaded_file is None else pd.read_csv(uploaded_file)
    
    # Convert transaction_date to datetime for filtering
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Risk Filters")
        
        min_date = df['transaction_date'].min().date()
        max_date = df['transaction_date'].max().date()
        date_range = st.date_input("Transaction Date Range", [min_date, max_date])
        
        min_amount, max_amount = st.slider(
            "Loan Amount Range", 
            min_value=0, 
            max_value=int(df['amount'].max() * 1.1),
            value=(0, int(df['amount'].max()))
        )
        
        risk_options = {
            "🚩 Unlicensed Platforms": "platform_license in ['NO-LICENSE', 'SUSPENDED']",
            "⚠️ Capital Adequacy < 8%": "platform_capital_ratio < 8",
            "🔻 Predatory Lending": "(interest_rate > 24) & (credit_score < 650)",
            "👥 Related Party Loans": "related_party_flag == 'Yes'",
            "📋 KYC Issues": "kyc_status != 'Verified'"
        }
        
        selected_risks = st.multiselect(
            "Risk Categories", 
            options=list(risk_options.keys()),
            default=list(risk_options.keys())
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if date_range:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) if len(date_range) > 1 else pd.Timestamp(date_range[0])
        filtered_df = filtered_df[
            (filtered_df['transaction_date'] >= start_date) & 
            (filtered_df['transaction_date'] <= end_date)
        ]
    
    filtered_df = filtered_df[
        (filtered_df['amount'] >= min_amount) & 
        (filtered_df['amount'] <= max_amount)
    ]
    
    if selected_risks:
        risk_query = " | ".join([risk_options[risk] for risk in selected_risks])
        filtered_df = filtered_df.query(risk_query)
    
    # Display data
    st.subheader("📊 Loan Portfolio Overview")
    st.dataframe(filtered_df.style.apply(highlight_risks, axis=1), height=400)
    
    # Dashboard
    st.subheader("📈 Risk Dashboard")
    create_dashboard(filtered_df)
    
    # LLM analysis section
    st.subheader("🤖 Deep Regulatory Scan")
    if st.button("Run AI Compliance Audit", type="primary"):
        with st.spinner("🔍 Scanning for regulatory violations..."):
            analysis_result = analyze_with_llm(filtered_df)
            st.success("Compliance audit completed!")
            
            # Display LLM results
            st.markdown("### AI Compliance Findings")
            st.markdown(analysis_result, unsafe_allow_html=True)
            
            # Download findings
            st.download_button(
                label="📥 Download Audit Report",
                data=analysis_result,
                file_name=f"compliance_audit_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()