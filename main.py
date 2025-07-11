import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize language settings
def get_translations():
    return {
        "en": {
            "title": "LOBSTERTECH",
            "subtitle": "Leveraging GPT-4o model trained to detect regulatory risks in peer-to-peer lending platforms",
            "upload_file": "Upload P2P Loan Data (CSV)",
            "loan_overview": "Loan Portfolio Overview",
            "risk_dashboard": "Risk Dashboard",
            "deep_scan": "Deep Regulatory Scan",
            "run_audit": "Run AI Compliance Audit",
            "download_report": "Download Audit Report",
            "filters": "🔧 Risk Filters",
            "date_range": "Transaction Date Range",
            "amount_range": "Loan Amount Range",
            "risk_categories": "Risk Categories",
            "scanning": "🔍 Scanning for regulatory violations...",
            "audit_complete": "Compliance audit completed!",
            "findings": "AI Compliance Findings",
        },
        "zh": {
            "title": "龙虾科技",
            "subtitle": "利用GPT-4o模型检测P2P借贷平台中的监管风险",
            "upload_file": "上传P2P贷款数据 (CSV)",
            "loan_overview": "贷款组合概览",
            "risk_dashboard": "风险仪表板",
            "deep_scan": "深度监管扫描",
            "run_audit": "运行AI合规审计",
            "download_report": "下载审计报告",
            "filters": "🔧 风险过滤器",
            "date_range": "交易日期范围",
            "amount_range": "贷款金额范围",
            "risk_categories": "风险类别",
            "scanning": "🔍 正在扫描监管违规行为...",
            "audit_complete": "合规审计完成！",
            "findings": "AI合规审查结果",
        }
    }

# Column name translations
COLUMN_TRANSLATIONS = {
    "platform_id": {"en": "Platform ID", "zh": "平台ID"},
    "loan_id": {"en": "Loan ID", "zh": "贷款ID"},
    "borrower_id": {"en": "Borrower ID", "zh": "借款人ID"},
    "amount": {"en": "Amount", "zh": "金额"},
    "interest_rate": {"en": "Interest Rate (%)", "zh": "利率 (%)"},
    "loan_term": {"en": "Loan Term (months)", "zh": "贷款期限 (月)"},
    "borrower_income": {"en": "Borrower Income", "zh": "借款人收入"},
    "credit_score": {"en": "Credit Score", "zh": "信用评分"},
    "loan_purpose": {"en": "Loan Purpose", "zh": "贷款用途"},
    "collateral_value": {"en": "Collateral Value", "zh": "抵押物价值"},
    "platform_license": {"en": "Platform License", "zh": "平台许可证"},
    "kyc_status": {"en": "KYC Status", "zh": "KYC状态"},
    "transaction_date": {"en": "Transaction Date", "zh": "交易日期"},
    "repayment_status": {"en": "Repayment Status", "zh": "还款状态"},
    "platform_capital_ratio": {"en": "Capital Ratio (%)", "zh": "资本充足率 (%)"},
    "related_party_flag": {"en": "Related Party", "zh": "关联方"},
}

# Value translations for categorical data
VALUE_TRANSLATIONS = {
    "Business expansion": {"en": "Business expansion", "zh": "业务扩展"},
    "Medical emergency": {"en": "Medical emergency", "zh": "医疗紧急情况"},
    "Home renovation": {"en": "Home renovation", "zh": "房屋装修"},
    "Debt consolidation": {"en": "Debt consolidation", "zh": "债务整合"},
    "Education loan": {"en": "Education loan", "zh": "教育贷款"},
    "Vacation": {"en": "Vacation", "zh": "度假"},
    "Verified": {"en": "Verified", "zh": "已验证"},
    "Expired": {"en": "Expired", "zh": "已过期"},
    "Pending": {"en": "Pending", "zh": "待处理"},
    "Delayed": {"en": "Delayed", "zh": "已延期"},
    "Defaulted": {"en": "Defaulted", "zh": "已违约"},
    "Current": {"en": "Current", "zh": "正常还款"},
    "Yes": {"en": "Yes", "zh": "是"},
    "No": {"en": "No", "zh": "否"},
    "LIC-A385": {"en": "LIC-A385", "zh": "许可证-A385"},
    "LIC-B441": {"en": "LIC-B441", "zh": "许可证-B441"},
    "LIC-C992": {"en": "LIC-C992", "zh": "许可证-C992"},
    "NO-LICENSE": {"en": "NO LICENSE", "zh": "无许可证"},
    "SUSPENDED": {"en": "SUSPENDED", "zh": "已暂停"},
}

# Predefined sets for value checking
NO_LICENSE_VALUES = {VALUE_TRANSLATIONS["NO-LICENSE"]["en"], VALUE_TRANSLATIONS["NO-LICENSE"]["zh"]}
SUSPENDED_VALUES = {VALUE_TRANSLATIONS["SUSPENDED"]["en"], VALUE_TRANSLATIONS["SUSPENDED"]["zh"]}
YES_VALUES = {VALUE_TRANSLATIONS["Yes"]["en"], VALUE_TRANSLATIONS["Yes"]["zh"]}
VERIFIED_VALUES = {VALUE_TRANSLATIONS["Verified"]["en"], VALUE_TRANSLATIONS["Verified"]["zh"]}

# Initialize API and model
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Generate sample data with bilingual support
def generate_sample_data(lang="en"):
    data = {
        "platform_id": ["P2P-001", "P2P-007", "P2P-003", "P2P-007", "P2P-005", "P2P-002"],
        "loan_id": ["L-1023", "L-5512", "L-9917", "L-3381", "L-4476", "L-8890"],
        "borrower_id": ["B-88921", "B-00238", "B-77432", "B-00991", "B-556123", "B-00238"],
        "amount": [50000, 200000, 80000, 350000, 120000, 150000],
        "interest_rate": [18.5, 24.9, 15.2, 32.5, 12.8, 28.5],
        "loan_term": [12, 24, 36, 6, 24, 3],
        "borrower_income": [120000, 65000, 185000, 42000, 95000, 68000],
        "credit_score": [715, 682, 781, 605, 698, 621],
        "loan_purpose": [
            VALUE_TRANSLATIONS["Business expansion"][lang],
            VALUE_TRANSLATIONS["Medical emergency"][lang],
            VALUE_TRANSLATIONS["Home renovation"][lang],
            VALUE_TRANSLATIONS["Debt consolidation"][lang],
            VALUE_TRANSLATIONS["Education loan"][lang],
            VALUE_TRANSLATIONS["Vacation"][lang],
        ],
        "collateral_value": [75000, 0, 120000, 0, 180000, 0],
        "platform_license": [
            VALUE_TRANSLATIONS["LIC-A385"][lang],
            VALUE_TRANSLATIONS["NO-LICENSE"][lang],
            VALUE_TRANSLATIONS["LIC-C992"][lang],
            VALUE_TRANSLATIONS["NO-LICENSE"][lang],
            VALUE_TRANSLATIONS["SUSPENDED"][lang],
            VALUE_TRANSLATIONS["LIC-B441"][lang],
        ],
        "kyc_status": [
            VALUE_TRANSLATIONS["Verified"][lang],
            VALUE_TRANSLATIONS["Expired"][lang],
            VALUE_TRANSLATIONS["Verified"][lang],
            VALUE_TRANSLATIONS["Pending"][lang],
            VALUE_TRANSLATIONS["Verified"][lang],
            VALUE_TRANSLATIONS["Verified"][lang],
        ],
        "transaction_date": ["2023-05-12", "2023-06-18", "2023-07-05", "2023-08-22", "2023-09-14", "2023-10-05"],
        "repayment_status": [
            VALUE_TRANSLATIONS["Delayed"][lang],
            VALUE_TRANSLATIONS["Defaulted"][lang],
            VALUE_TRANSLATIONS["Current"][lang],
            VALUE_TRANSLATIONS["Delayed"][lang],
            VALUE_TRANSLATIONS["Current"][lang],
            VALUE_TRANSLATIONS["Current"][lang],
        ],
        "platform_capital_ratio": [8.2, 3.1, 12.5, 2.8, 6.7, 9.2],
        "related_party_flag": [
            VALUE_TRANSLATIONS["No"][lang],
            VALUE_TRANSLATIONS["Yes"][lang],
            VALUE_TRANSLATIONS["No"][lang],
            VALUE_TRANSLATIONS["Yes"][lang],
            VALUE_TRANSLATIONS["No"][lang],
            VALUE_TRANSLATIONS["No"][lang],
        ],
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

# Robust risk highlighting that works with both languages
def highlight_risks(row):
    styles = [''] * len(row)
    
    # Get column names
    col_names = row.index.tolist()
    
    # Find column indices dynamically
    platform_license_col = next((i for i, name in enumerate(col_names) 
                              if "license" in name.lower() or "许可证" in name), None)
    interest_rate_col = next((i for i, name in enumerate(col_names) 
                           if "interest" in name.lower() or "利率" in name), None)
    credit_score_col = next((i for i, name in enumerate(col_names) 
                           if "credit" in name.lower() or "信用" in name), None)
    capital_ratio_col = next((i for i, name in enumerate(col_names) 
                           if "capital" in name.lower() or "资本" in name), None)
    related_party_col = next((i for i, name in enumerate(col_names) 
                            if "related" in name.lower() or "关联" in name), None)
    kyc_status_col = next((i for i, name in enumerate(col_names) 
                         if "kyc" in name.lower() or "kyc" in name), None)

    # Unlicensed platform
    if platform_license_col is not None:
        license_value = row.iloc[platform_license_col]
        if license_value in NO_LICENSE_VALUES or license_value in SUSPENDED_VALUES:
            styles[platform_license_col] = 'background-color: #FFCCCB'
        
    # Capital adequacy
    if capital_ratio_col is not None and row.iloc[capital_ratio_col] < 8:
        styles[capital_ratio_col] = 'background-color: #FFD580'
        
    # Predatory lending
    if (interest_rate_col is not None and 
        credit_score_col is not None and
        row.iloc[interest_rate_col] > 24 and 
        row.iloc[credit_score_col] < 650):
        styles[interest_rate_col] = 'background-color: #FF6666'
        styles[credit_score_col] = 'background-color: #FF6666'
        
    # Related party
    if related_party_col is not None:
        party_value = row.iloc[related_party_col]
        if party_value in YES_VALUES:
            styles[related_party_col] = 'background-color: #90EE90'
        
    # KYC issues
    if kyc_status_col is not None:
        kyc_value = row.iloc[kyc_status_col]
        if kyc_value not in VERIFIED_VALUES:
            styles[kyc_status_col] = 'background-color: #ADD8E6'
        
    return styles

# Dashboard visualization
def create_dashboard(df, lang="en"):
    tab1, tab2, tab3 = st.tabs([
        f"{'License Compliance' if lang == 'en' else '许可证合规性'}",
        f"{'Risk Exposure' if lang == 'en' else '风险敞口'}",
        f"{'Borrower Analysis' if lang == 'en' else '借款人分析'}"
    ])
    
    with tab1:
        st.subheader("Platform License Status" if lang == "en" else "平台许可证状态")
        license_counts = df['platform_license'].apply(
            lambda x: "Valid" if "LIC" in x else "Invalid"
        ).value_counts()
        if lang == "zh":
            license_counts.index = license_counts.index.map({"Valid": "有效", "Invalid": "无效"})
        st.bar_chart(license_counts)
        
    with tab2:
        st.subheader("Platform Risk Exposure" if lang == "en" else "平台风险敞口")
        conditions = [
            (df['interest_rate'] > 24) & (df['credit_score'] < 650),
            df['platform_license'].isin([
                VALUE_TRANSLATIONS["NO-LICENSE"]["en"], 
                VALUE_TRANSLATIONS["SUSPENDED"]["en"]
            ]),
            df['platform_capital_ratio'] < 8
        ]
        choices = [3, 2, 1]
        df['risk_score'] = np.select(conditions, choices, default=0)
        platform_risk = df.groupby('platform_id')['risk_score'].max().sort_values()
        st.bar_chart(platform_risk)
        
    with tab3:
        st.subheader("Credit Score vs Interest Rate" if lang == "en" else "信用评分与利率")
        chart_data = df[['credit_score', 'interest_rate', 'amount', 'repayment_status']].copy()
        chart_data['size'] = chart_data['amount'] / 10000
        st.scatter_chart(
            chart_data,
            x='credit_score',
            y='interest_rate',
            size='size',
            color='repayment_status'
        )

# Main app
def main():
    st.set_page_config(
        page_title="P2P Lending RegTech Monitor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Language selection
    lang = st.sidebar.radio("Language/语言", ["English", "中文"], index=0)
    lang_key = "en" if lang == "English" else "zh"
    trans = get_translations()[lang_key]
    
    st.title(trans["title"])
    st.caption(trans["subtitle"])
    
    # Data upload
    uploaded_file = st.file_uploader(trans["upload_file"], type="csv")
    df = generate_sample_data(lang_key) if uploaded_file is None else pd.read_csv(uploaded_file)
    
    # Convert transaction_date to datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Sidebar filters
    with st.sidebar:
        st.header(trans["filters"])
        
        min_date = df['transaction_date'].min().date()
        max_date = df['transaction_date'].max().date()
        date_range = st.date_input(
            trans["date_range"], 
            [min_date, max_date]
        )
        
        min_amount, max_amount = st.slider(
            trans["amount_range"], 
            min_value=0, 
            max_value=int(df['amount'].max() * 1.1),
            value=(0, int(df['amount'].max()))
        )
        
        risk_options = {
            "🚩 Unlicensed Platforms" if lang_key == "en" else "🚩 无证平台": "platform_license in ['NO-LICENSE', 'SUSPENDED']",
            "⚠️ Capital Adequacy < 8%" if lang_key == "en" else "⚠️ 资本充足率 < 8%": "platform_capital_ratio < 8",
            "🔻 Predatory Lending" if lang_key == "en" else "🔻 掠夺性贷款": "(interest_rate > 24) & (credit_score < 650)",
            "👥 Related Party Loans" if lang_key == "en" else "👥 关联方贷款": "related_party_flag == 'Yes'",
            "📋 KYC Issues" if lang_key == "en" else "📋 KYC问题": "kyc_status != 'Verified'"
        }
        
        selected_risks = st.multiselect(
            trans["risk_categories"], 
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
    
    # Display data with translated column names
    st.subheader(trans["loan_overview"])
    
    # Apply styling before translating column names
    styled_df = filtered_df.style.apply(highlight_risks, axis=1)
    
    # Translate column names for display
    display_df = filtered_df.copy()
    display_df.columns = [COLUMN_TRANSLATIONS.get(col, {}).get(lang_key, col) 
                         for col in display_df.columns]
    
    # Display the styled DataFrame with translated columns
    st.dataframe(styled_df.set_columns(display_df.columns), height=400)
    
    # Dashboard
    st.subheader(trans["risk_dashboard"])
    create_dashboard(filtered_df, lang_key)
    
    # LLM analysis section
    st.subheader(trans["deep_scan"])
    if st.button(trans["run_audit"], type="primary"):
        with st.spinner(trans["scanning"]):
            analysis_result = analyze_with_llm(filtered_df)
            st.success(trans["audit_complete"])
            
            st.markdown(f"### {trans['findings']}")
            st.markdown(analysis_result, unsafe_allow_html=True)
            
            st.download_button(
                label=trans["download_report"],
                data=analysis_result,
                file_name=f"compliance_audit_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()