import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import plotly.express as px

api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key
# Initialize LangChain model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Lobster-themed styling
def set_lobster_theme():
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #fff0f0 0%, #fff9f0 100%);
        background-attachment: fixed;
    }
    
    /* Lobster-themed headers */
    h1, h2, h3, h4, h5 {
        color: #d32f2f !important;
        font-family: 'Arial Rounded MT Bold', sans-serif;
        border-bottom: 2px dashed #ff8a80;
        padding-bottom: 5px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #ffebee 0%, #ffecb3 100%);
        border-right: 3px solid #ff8a80;
    }
    
    /* Lobster button styling */
    .stButton>button {
        background: linear-gradient(135deg, #ff5252 0%, #ff8a80 100%);
        color: white !important;
        border-radius: 20px;
        border: none;
        box-shadow: 0 4px 8px rgba(211, 47, 47, 0.3);
        transition: all 0.3s;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #d32f2f 0%, #ff5252 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(211, 47, 47, 0.4);
    }
    
    /* Lobster tabs */
    [data-baseweb="tab-list"] {
        background-color: #ffebee !important;
        border-radius: 10px;
        padding: 5px;
    }
    
    [data-baseweb="tab"] {
        border-radius: 10px !important;
        margin: 0 5px !important;
        transition: all 0.3s !important;
    }
    
    [aria-selected="true"] {
        background-color: #ff8a80 !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    /* Lobster cards */
    .stDataFrame {
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(211, 47, 47, 0.2);
    }
    
    /* Lobster progress spinner */
    .stSpinner > div > div {
        border-top-color: #ff5252 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Lobster images in base64 format
def get_lobster_image():
    return "https://images.unsplash.com/photo-1588345921523-c2dcdb7f1dcd?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80"

def get_lobster_icon():
    return "https://cdn-icons-png.flaticon.com/512/4271/4271898.png"

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

# Dashboard visualization
def create_dashboard(df):
    tab1, tab2, tab3 = st.tabs(["🦞 License Compliance 许可证合规", "🦞 Risk Exposure 风险敞口", "🦞 Borrower Analysis 借款人分析"])
    
    with tab1:
        st.subheader("Platform License Status 平台许可证状态")
        license_counts = df['platform_license'].apply(
            lambda x: "Valid 有效" if x.startswith('LIC') else "Invalid 无效"
        ).value_counts()
        
        fig = px.pie(
            license_counts, 
            values=license_counts.values, 
            names=license_counts.index,
            color=license_counts.index,
            color_discrete_map={'Valid 有效': '#4CAF50', 'Invalid 无效': '#F44336'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Platform Risk Exposure 平台风险敞口")
        # Calculate risk scores
        conditions = [
            (df['interest_rate'] > 24) & (df['credit_score'] < 650),
            df['platform_license'].isin(['NO-LICENSE', 'SUSPENDED']),
            df['platform_capital_ratio'] < 8
        ]
        choices = [3, 2, 1]
        df['risk_score'] = np.select(conditions, choices, default=0)
        
        # Aggregate risk by platform
        platform_risk = df.groupby('platform_id')['risk_score'].max().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            platform_risk, 
            x='platform_id', 
            y='risk_score',
            color='risk_score',
            color_continuous_scale='Reds',
            text='risk_score',
            labels={'risk_score': 'Risk Score', 'platform_id': 'Platform ID'}
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(yaxis_title='Risk Score', xaxis_title='Platform ID')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.subheader("Credit Score vs Interest Rate 信用评分 vs 利率")
        # Use Plotly scatter chart
        fig = px.scatter(
            df, 
            x='credit_score', 
            y='interest_rate',
            size='amount',
            color='repayment_status',
            hover_name='loan_id',
            color_discrete_map={
                'Current': '#4CAF50',
                'Delayed': '#FFC107',
                'Defaulted': '#F44336'
            },
            size_max=30,
            labels={
                'credit_score': 'Credit Score',
                'interest_rate': 'Interest Rate (%)',
                'amount': 'Loan Amount',
                'repayment_status': 'Repayment Status'
            }
        )
        fig.update_layout(
            xaxis=dict(range=[500, 850]),
            yaxis=dict(range=[10, 35])
        )
        st.plotly_chart(fig, use_container_width=True)

# Streamlit app
def main():
    set_lobster_theme()
    
    st.set_page_config(
        page_title="P2P Lending RegTech Monitor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with lobster image
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(get_lobster_icon(), width=100)
    with col2:
        st.title("🦞 LOBSTER POLICE 监管龙虾")
        st.caption("Leveraging GPT-4o to detect regulatory risks in peer-to-peer lending platforms / 利用 GPT-4o 检测点对点借贷平台中的监管风险")
    
    # Lobster divider
    st.image(get_lobster_image(), use_column_width=True)
    
    # Data upload section
    st.subheader("📥 Data Management 数据管理")
    uploaded_file = st.file_uploader("Upload P2P Loan Data (CSV) 上传P2P贷款数据(CSV)", type="csv")
    df = generate_sample_data() if uploaded_file is None else pd.read_csv(uploaded_file)
    
    # Convert transaction_date to datetime for filtering
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Sidebar filters
    with st.sidebar:
        # Lobster sidebar header
        st.image(get_lobster_icon(), width=80)
        st.header("🔧 Risk Filters 风险过滤器")
        
        # Date filter
        min_date = df['transaction_date'].min().date()
        max_date = df['transaction_date'].max().date()
        date_range = st.date_input("Transaction Date Range 交易日期范围", [min_date, max_date])
        
        # Amount filter
        min_amount, max_amount = st.slider(
            "Loan Amount Range 贷款金额范围 (USD)", 
            min_value=0, 
            max_value=int(df['amount'].max() * 1.1),
            value=(0, int(df['amount'].max()))
        )
        
        # Risk filters
        risk_options = {
            "🚩 Unlicensed Platforms 无证平台": "platform_license in ['NO-LICENSE', 'SUSPENDED']",
            "⚠️ Capital Adequacy < 8% 资本充足率<8%": "platform_capital_ratio < 8",
            "🔻 Predatory Lending 掠夺性贷款": "(interest_rate > 24) & (credit_score < 650)",
            "👥 Related Party Loans 关联方贷款": "related_party_flag == 'Yes'",
            "📋 KYC Issues KYC问题": "kyc_status != 'Verified'"
        }
        
        selected_risks = st.multiselect(
            "Risk Categories 风险类别", 
            options=list(risk_options.keys()),
            default=list(risk_options.keys())
        )
        
        # Lobster facts for fun
        st.divider()
        st.subheader("🦞 Lobster Facts")
        lobster_facts = [
            "Lobsters can live up to 100 years!",
            "Lobsters have a dominant right or left claw",
            "Lobsters taste with their legs!",
            "Lobsters molt about 25 times in their first 5 years",
            "Lobsters keep growing throughout their lives"
        ]
        st.info(f"**Did you know?**\n\n{np.random.choice(lobster_facts)}")
    
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
    st.subheader("📊 Loan Portfolio Overview 贷款组合概览")
    st.dataframe(filtered_df.style.apply(highlight_risks, axis=1), height=400)
    
    # Summary stats
    st.subheader("📈 Portfolio Summary 投资组合摘要")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Loans 总贷款数", len(filtered_df), f"{len(filtered_df)/len(df):.0%} of total")
    with col2:
        st.metric("Total Value 总价值", f"${filtered_df['amount'].sum():,}")
    with col3:
        avg_rate = filtered_df['interest_rate'].mean()
        st.metric("Avg Interest Rate 平均利率", f"{avg_rate:.1f}%", 
                 "High Risk" if avg_rate > 24 else "Normal" if avg_rate > 15 else "Low")
    with col4:
        avg_score = filtered_df['credit_score'].mean()
        st.metric("Avg Credit Score 平均信用评分", f"{avg_score:.0f}", 
                 "High Risk" if avg_score < 650 else "Good" if avg_score > 700 else "Fair")
    
    # Dashboard
    st.subheader("📊 Risk Dashboard 风险仪表板")
    create_dashboard(filtered_df)
    
    # LLM analysis section
    st.subheader("🤖 Deep Regulatory Scan 深度监管扫描")
    st.info("Our AI-powered Lobster Police system will scan your portfolio for regulatory violations and compliance risks")
    
    if st.button("🦞 Run AI Compliance Audit 运行AI合规审计", type="primary", use_container_width=True):
        with st.spinner("🔍 Scanning for regulatory violations... 正在扫描违规行为..."):
            analysis_result = analyze_with_llm(filtered_df)
            st.success("Compliance audit completed! 合规审计完成！")
            
            # Display LLM results
            st.markdown("### 🦞 AI Compliance Findings AI合规发现")
            st.markdown(analysis_result, unsafe_allow_html=True)
            
            # Download findings
            st.download_button(
                label="📥 Download Audit Report 下载审计报告",
                data=analysis_result,
                file_name=f"lobster_audit_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()