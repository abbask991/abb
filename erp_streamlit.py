import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise ERP",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label { font-size: 0.85rem; color: #6b7280; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    .status-active { background: #d1fae5; color: #065f46; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .status-pending { background: #fef3c7; color: #92400e; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .status-danger { background: #fee2e2; color: #991b1b; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .status-info { background: #dbeafe; color: #1e40af; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    h1 { font-size: 1.8rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.3rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #d1d5db !important; }
    [data-testid="stSidebar"] .stRadio label:hover { color: white !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Mock Data
# ──────────────────────────────────────────────

def get_employees():
    return pd.DataFrame([
        {"ID": "E001", "Name": "Sarah Chen", "Department": "Engineering", "Position": "VP of Engineering", "Status": "Active", "Hire Date": "2019-03-15", "Salary": 185000, "Email": "sarah.chen@company.com", "Phone": "+1-555-0101"},
        {"ID": "E002", "Name": "James Miller", "Department": "Sales", "Position": "VP of Sales", "Status": "Active", "Hire Date": "2019-06-01", "Salary": 170000, "Email": "james.miller@company.com", "Phone": "+1-555-0102"},
        {"ID": "E003", "Name": "Emily Davis", "Department": "Marketing", "Position": "Marketing Director", "Status": "Active", "Hire Date": "2020-01-10", "Salary": 145000, "Email": "emily.davis@company.com", "Phone": "+1-555-0103"},
        {"ID": "E004", "Name": "Maria Garcia", "Department": "Human Resources", "Position": "HR Director", "Status": "Active", "Hire Date": "2019-08-20", "Salary": 130000, "Email": "maria.garcia@company.com", "Phone": "+1-555-0104"},
        {"ID": "E005", "Name": "Robert Kim", "Department": "Finance", "Position": "CFO", "Status": "Active", "Hire Date": "2018-11-05", "Salary": 195000, "Email": "robert.kim@company.com", "Phone": "+1-555-0105"},
        {"ID": "E006", "Name": "John Park", "Department": "Engineering", "Position": "Senior Developer", "Status": "Active", "Hire Date": "2020-04-12", "Salary": 135000, "Email": "john.park@company.com", "Phone": "+1-555-0106"},
        {"ID": "E007", "Name": "Lisa Brown", "Department": "Customer Support", "Position": "Support Manager", "Status": "Active", "Hire Date": "2020-07-18", "Salary": 95000, "Email": "lisa.brown@company.com", "Phone": "+1-555-0107"},
        {"ID": "E008", "Name": "David Wilson", "Department": "Operations", "Position": "COO", "Status": "Active", "Hire Date": "2019-02-01", "Salary": 190000, "Email": "david.wilson@company.com", "Phone": "+1-555-0108"},
        {"ID": "E009", "Name": "Alex Turner", "Department": "Product", "Position": "Head of Product", "Status": "Active", "Hire Date": "2020-09-01", "Salary": 155000, "Email": "alex.turner@company.com", "Phone": "+1-555-0109"},
        {"ID": "E010", "Name": "Rachel Lee", "Department": "Engineering", "Position": "Frontend Lead", "Status": "Active", "Hire Date": "2021-01-15", "Salary": 125000, "Email": "rachel.lee@company.com", "Phone": "+1-555-0110"},
        {"ID": "E011", "Name": "Michael Torres", "Department": "Sales", "Position": "Account Executive", "Status": "Active", "Hire Date": "2021-03-20", "Salary": 85000, "Email": "michael.torres@company.com", "Phone": "+1-555-0111"},
        {"ID": "E012", "Name": "Sophia Nguyen", "Department": "Marketing", "Position": "Content Strategist", "Status": "On Leave", "Hire Date": "2021-06-10", "Salary": 78000, "Email": "sophia.nguyen@company.com", "Phone": "+1-555-0112"},
        {"ID": "E013", "Name": "Daniel Johnson", "Department": "Engineering", "Position": "Backend Developer", "Status": "Active", "Hire Date": "2021-08-05", "Salary": 115000, "Email": "daniel.johnson@company.com", "Phone": "+1-555-0113"},
        {"ID": "E014", "Name": "Amanda White", "Department": "Finance", "Position": "Senior Accountant", "Status": "Active", "Hire Date": "2020-11-15", "Salary": 95000, "Email": "amanda.white@company.com", "Phone": "+1-555-0114"},
        {"ID": "E015", "Name": "Chris Anderson", "Department": "Operations", "Position": "Logistics Manager", "Status": "Active", "Hire Date": "2021-02-28", "Salary": 88000, "Email": "chris.anderson@company.com", "Phone": "+1-555-0115"},
        {"ID": "E016", "Name": "Jessica Taylor", "Department": "Sales", "Position": "Sales Manager", "Status": "Active", "Hire Date": "2020-05-15", "Salary": 110000, "Email": "jessica.taylor@company.com", "Phone": "+1-555-0116"},
        {"ID": "E017", "Name": "Brian Harris", "Department": "Engineering", "Position": "DevOps Engineer", "Status": "Active", "Hire Date": "2021-10-01", "Salary": 120000, "Email": "brian.harris@company.com", "Phone": "+1-555-0117"},
        {"ID": "E018", "Name": "Olivia Martinez", "Department": "Human Resources", "Position": "Recruiter", "Status": "Active", "Hire Date": "2022-01-10", "Salary": 72000, "Email": "olivia.martinez@company.com", "Phone": "+1-555-0118"},
        {"ID": "E019", "Name": "Kevin Wright", "Department": "Customer Support", "Position": "Support Agent", "Status": "Terminated", "Hire Date": "2021-04-15", "Salary": 55000, "Email": "kevin.wright@company.com", "Phone": "+1-555-0119"},
        {"ID": "E020", "Name": "Nina Patel", "Department": "Product", "Position": "Product Manager", "Status": "Active", "Hire Date": "2022-03-01", "Salary": 120000, "Email": "nina.patel@company.com", "Phone": "+1-555-0120"},
    ])

def get_products():
    return pd.DataFrame([
        {"ID": "P001", "SKU": "WDG-001", "Name": "Industrial Widget A", "Category": "Widgets", "Price": 29.99, "Cost": 12.50, "Stock": 2450, "Reorder Level": 500, "Warehouse": "Main Warehouse", "Status": "In Stock"},
        {"ID": "P002", "SKU": "WDG-002", "Name": "Premium Widget B", "Category": "Widgets", "Price": 49.99, "Cost": 22.00, "Stock": 180, "Reorder Level": 200, "Warehouse": "Main Warehouse", "Status": "Low Stock"},
        {"ID": "P003", "SKU": "GDG-001", "Name": "Gadget Pro X1", "Category": "Gadgets", "Price": 199.99, "Cost": 85.00, "Stock": 890, "Reorder Level": 100, "Warehouse": "East Distribution", "Status": "In Stock"},
        {"ID": "P004", "SKU": "GDG-002", "Name": "Gadget Mini S2", "Category": "Gadgets", "Price": 79.99, "Cost": 35.00, "Stock": 0, "Reorder Level": 150, "Warehouse": "Main Warehouse", "Status": "Out of Stock"},
        {"ID": "P005", "SKU": "CMP-001", "Name": "Component Alpha", "Category": "Components", "Price": 8.50, "Cost": 3.20, "Stock": 12000, "Reorder Level": 2000, "Warehouse": "Main Warehouse", "Status": "In Stock"},
        {"ID": "P006", "SKU": "CMP-002", "Name": "Component Beta", "Category": "Components", "Price": 12.75, "Cost": 5.80, "Stock": 8500, "Reorder Level": 1500, "Warehouse": "West Facility", "Status": "In Stock"},
        {"ID": "P007", "SKU": "ASM-001", "Name": "Assembly Kit Pro", "Category": "Assemblies", "Price": 349.99, "Cost": 150.00, "Stock": 45, "Reorder Level": 50, "Warehouse": "East Distribution", "Status": "Low Stock"},
        {"ID": "P008", "SKU": "ASM-002", "Name": "Assembly Kit Standard", "Category": "Assemblies", "Price": 189.99, "Cost": 80.00, "Stock": 320, "Reorder Level": 75, "Warehouse": "Main Warehouse", "Status": "In Stock"},
        {"ID": "P009", "SKU": "TLS-001", "Name": "Precision Tool Set", "Category": "Tools", "Price": 599.99, "Cost": 280.00, "Stock": 65, "Reorder Level": 20, "Warehouse": "West Facility", "Status": "In Stock"},
        {"ID": "P010", "SKU": "TLS-002", "Name": "Safety Equipment Pack", "Category": "Safety", "Price": 129.99, "Cost": 55.00, "Stock": 410, "Reorder Level": 100, "Warehouse": "Main Warehouse", "Status": "In Stock"},
        {"ID": "P011", "SKU": "RAW-001", "Name": "Steel Rods (Bundle)", "Category": "Raw Materials", "Price": 89.99, "Cost": 42.00, "Stock": 1800, "Reorder Level": 300, "Warehouse": "West Facility", "Status": "In Stock"},
        {"ID": "P012", "SKU": "RAW-002", "Name": "Copper Wire (Spool)", "Category": "Raw Materials", "Price": 45.00, "Cost": 20.00, "Stock": 95, "Reorder Level": 100, "Warehouse": "Main Warehouse", "Status": "Low Stock"},
    ])

def get_sales_orders():
    return pd.DataFrame([
        {"Order #": "SO-2026-089", "Customer": "Acme Corporation", "Items": 2, "Total": 17277.30, "Order Date": "2026-05-01", "Status": "Delivered", "Payment": "Paid"},
        {"Order #": "SO-2026-090", "Customer": "Global Dynamics", "Items": 2, "Total": 7019.84, "Order Date": "2026-05-10", "Status": "Shipped", "Payment": "Paid"},
        {"Order #": "SO-2026-091", "Customer": "Summit Industries", "Items": 1, "Total": 45900.00, "Order Date": "2026-05-15", "Status": "Confirmed", "Payment": "Unpaid"},
        {"Order #": "SO-2026-092", "Customer": "TechFlow Inc", "Items": 2, "Total": 9611.41, "Order Date": "2026-05-16", "Status": "Confirmed", "Payment": "Partial"},
        {"Order #": "SO-2026-093", "Customer": "Pinnacle Group", "Items": 1, "Total": 20518.92, "Order Date": "2026-05-18", "Status": "Draft", "Payment": "Unpaid"},
        {"Order #": "SO-2026-094", "Customer": "Pacific Trade Co", "Items": 2, "Total": 32399.46, "Order Date": "2026-05-08", "Status": "Shipped", "Payment": "Paid"},
    ])

def get_customers():
    return pd.DataFrame([
        {"ID": "C001", "Name": "John Bradley", "Company": "Acme Corporation", "Segment": "Enterprise", "Total Orders": 47, "Total Spent": 284500, "Status": "Active", "Last Order": "2026-05-14"},
        {"ID": "C002", "Name": "Sarah Mitchell", "Company": "TechFlow Inc", "Segment": "Mid-Market", "Total Orders": 32, "Total Spent": 156200, "Status": "Active", "Last Order": "2026-05-10"},
        {"ID": "C003", "Name": "Robert Hayes", "Company": "Global Dynamics", "Segment": "Enterprise", "Total Orders": 65, "Total Spent": 532800, "Status": "Active", "Last Order": "2026-05-18"},
        {"ID": "C004", "Name": "Jennifer Walsh", "Company": "StartUp Hub", "Segment": "Small Business", "Total Orders": 12, "Total Spent": 34500, "Status": "Active", "Last Order": "2026-04-28"},
        {"ID": "C005", "Name": "Michael Chen", "Company": "Pacific Trade Co", "Segment": "Mid-Market", "Total Orders": 28, "Total Spent": 198700, "Status": "Active", "Last Order": "2026-05-08"},
        {"ID": "C006", "Name": "Laura Bennett", "Company": "Summit Industries", "Segment": "Enterprise", "Total Orders": 41, "Total Spent": 312400, "Status": "Active", "Last Order": "2026-05-16"},
        {"ID": "C007", "Name": "Thomas Reid", "Company": "Micro Solutions", "Segment": "Small Business", "Total Orders": 8, "Total Spent": 22100, "Status": "Inactive", "Last Order": "2025-12-15"},
        {"ID": "C008", "Name": "Angela Foster", "Company": "Pinnacle Group", "Segment": "Enterprise", "Total Orders": 55, "Total Spent": 445600, "Status": "Active", "Last Order": "2026-05-12"},
    ])

def get_pipeline():
    return pd.DataFrame([
        {"Deal": "Acme Q3 Expansion", "Customer": "Acme Corporation", "Value": 125000, "Stage": "Negotiation", "Probability": 75, "Expected Close": "2026-06-30", "Assigned To": "James Miller"},
        {"Deal": "TechFlow Annual Contract", "Customer": "TechFlow Inc", "Value": 85000, "Stage": "Proposal", "Probability": 60, "Expected Close": "2026-07-15", "Assigned To": "Jessica Taylor"},
        {"Deal": "Global Dynamics Enterprise", "Customer": "Global Dynamics", "Value": 310000, "Stage": "Qualified", "Probability": 40, "Expected Close": "2026-08-01", "Assigned To": "James Miller"},
        {"Deal": "Summit Manufacturing Deal", "Customer": "Summit Industries", "Value": 175000, "Stage": "Closed Won", "Probability": 100, "Expected Close": "2026-05-15", "Assigned To": "Michael Torres"},
        {"Deal": "Pinnacle New Line", "Customer": "Pinnacle Group", "Value": 220000, "Stage": "Negotiation", "Probability": 70, "Expected Close": "2026-06-15", "Assigned To": "Jessica Taylor"},
        {"Deal": "Pacific Trade Renewal", "Customer": "Pacific Trade Co", "Value": 95000, "Stage": "Proposal", "Probability": 55, "Expected Close": "2026-07-01", "Assigned To": "Michael Torres"},
        {"Deal": "StartUp Hub Pilot", "Customer": "StartUp Hub", "Value": 15000, "Stage": "Lead", "Probability": 20, "Expected Close": "2026-09-01", "Assigned To": "Michael Torres"},
        {"Deal": "Micro Solutions Upsell", "Customer": "Micro Solutions", "Value": 42000, "Stage": "Closed Lost", "Probability": 0, "Expected Close": "2026-04-30", "Assigned To": "Jessica Taylor"},
    ])

def get_suppliers():
    return pd.DataFrame([
        {"ID": "S001", "Name": "SteelWorks Global", "Contact": "Han Wei", "Rating": 4.8, "Total Orders": 120, "Status": "Active", "Payment Terms": "Net 30", "Lead Time": "14 days"},
        {"ID": "S002", "Name": "Component Direct", "Contact": "Patricia Owens", "Rating": 4.5, "Total Orders": 85, "Status": "Active", "Payment Terms": "Net 45", "Lead Time": "7 days"},
        {"ID": "S003", "Name": "TechParts Co", "Contact": "Greg Nelson", "Rating": 4.2, "Total Orders": 62, "Status": "Active", "Payment Terms": "Net 30", "Lead Time": "10 days"},
        {"ID": "S004", "Name": "Pacific Materials", "Contact": "Yuki Tanaka", "Rating": 3.9, "Total Orders": 38, "Status": "Active", "Payment Terms": "Net 60", "Lead Time": "21 days"},
        {"ID": "S005", "Name": "Atlas Safety Supplies", "Contact": "Mark Fischer", "Rating": 4.6, "Total Orders": 55, "Status": "Active", "Payment Terms": "Net 30", "Lead Time": "5 days"},
        {"ID": "S006", "Name": "QuickShip Electronics", "Contact": "Diana Flores", "Rating": 3.5, "Total Orders": 22, "Status": "Inactive", "Payment Terms": "Net 15", "Lead Time": "3 days"},
    ])

def get_purchase_orders():
    return pd.DataFrame([
        {"PO #": "PO-2026-042", "Supplier": "SteelWorks Global", "Items": 1, "Total": 22680, "Order Date": "2026-04-25", "Expected": "2026-05-09", "Status": "Received"},
        {"PO #": "PO-2026-043", "Supplier": "Component Direct", "Items": 2, "Total": 36072, "Order Date": "2026-05-01", "Expected": "2026-05-08", "Status": "Received"},
        {"PO #": "PO-2026-044", "Supplier": "TechParts Co", "Items": 1, "Total": 18360, "Order Date": "2026-05-12", "Expected": "2026-05-22", "Status": "Confirmed"},
        {"PO #": "PO-2026-045", "Supplier": "Component Direct", "Items": 1, "Total": 10368, "Order Date": "2026-05-15", "Expected": "2026-05-22", "Status": "Sent"},
        {"PO #": "PO-2026-046", "Supplier": "Atlas Safety Supplies", "Items": 1, "Total": 11880, "Order Date": "2026-05-18", "Expected": "2026-05-23", "Status": "Draft"},
    ])

def get_accounts():
    return pd.DataFrame([
        {"Code": "1000", "Name": "Cash and Cash Equivalents", "Type": "Asset", "Balance": 1245000},
        {"Code": "1100", "Name": "Accounts Receivable", "Type": "Asset", "Balance": 385200},
        {"Code": "1200", "Name": "Inventory", "Type": "Asset", "Balance": 892400},
        {"Code": "1300", "Name": "Prepaid Expenses", "Type": "Asset", "Balance": 45000},
        {"Code": "1500", "Name": "Fixed Assets", "Type": "Asset", "Balance": 2100000},
        {"Code": "1510", "Name": "Accumulated Depreciation", "Type": "Asset", "Balance": -420000},
        {"Code": "2000", "Name": "Accounts Payable", "Type": "Liability", "Balance": 198500},
        {"Code": "2100", "Name": "Accrued Liabilities", "Type": "Liability", "Balance": 125000},
        {"Code": "2200", "Name": "Short-term Debt", "Type": "Liability", "Balance": 300000},
        {"Code": "2500", "Name": "Long-term Debt", "Type": "Liability", "Balance": 750000},
        {"Code": "3000", "Name": "Common Stock", "Type": "Equity", "Balance": 1500000},
        {"Code": "3100", "Name": "Retained Earnings", "Type": "Equity", "Balance": 874100},
        {"Code": "4000", "Name": "Product Revenue", "Type": "Revenue", "Balance": 4250000},
        {"Code": "4100", "Name": "Service Revenue", "Type": "Revenue", "Balance": 850000},
        {"Code": "5000", "Name": "Cost of Goods Sold", "Type": "Expense", "Balance": 2125000},
        {"Code": "6000", "Name": "Salaries & Wages", "Type": "Expense", "Balance": 1450000},
        {"Code": "6100", "Name": "Rent & Utilities", "Type": "Expense", "Balance": 180000},
        {"Code": "6200", "Name": "Marketing & Advertising", "Type": "Expense", "Balance": 320000},
        {"Code": "6300", "Name": "Office Supplies & Equipment", "Type": "Expense", "Balance": 75000},
        {"Code": "6400", "Name": "Insurance", "Type": "Expense", "Balance": 48000},
    ])

def get_invoices():
    return pd.DataFrame([
        {"Invoice #": "INV-2026-0321", "Customer": "Acme Corporation", "Issue Date": "2026-05-01", "Due Date": "2026-05-31", "Total": 17277.30, "Paid": 17277.30, "Balance": 0.00, "Status": "Paid"},
        {"Invoice #": "INV-2026-0322", "Customer": "Global Dynamics", "Issue Date": "2026-05-10", "Due Date": "2026-06-09", "Total": 7019.84, "Paid": 7019.84, "Balance": 0.00, "Status": "Paid"},
        {"Invoice #": "INV-2026-0323", "Customer": "Summit Industries", "Issue Date": "2026-05-15", "Due Date": "2026-06-14", "Total": 45900.00, "Paid": 0.00, "Balance": 45900.00, "Status": "Sent"},
        {"Invoice #": "INV-2026-0324", "Customer": "TechFlow Inc", "Issue Date": "2026-05-16", "Due Date": "2026-06-15", "Total": 9611.41, "Paid": 5000.00, "Balance": 4611.41, "Status": "Sent"},
        {"Invoice #": "INV-2026-0325", "Customer": "Pinnacle Group", "Issue Date": "2026-04-01", "Due Date": "2026-04-30", "Total": 10259.46, "Paid": 0.00, "Balance": 10259.46, "Status": "Overdue"},
    ])

def get_projects():
    return pd.DataFrame([
        {"Name": "ERP System Implementation", "Client": "Internal", "Manager": "Alex Turner", "Status": "Active", "Priority": "High", "Start": "2026-01-15", "End": "2026-09-30", "Budget": 450000, "Spent": 185000, "Progress": 42},
        {"Name": "Website Redesign", "Client": "Internal", "Manager": "Emily Davis", "Status": "Active", "Priority": "Medium", "Start": "2026-03-01", "End": "2026-07-15", "Budget": 120000, "Spent": 78000, "Progress": 65},
        {"Name": "Acme Corp Integration", "Client": "Acme Corporation", "Manager": "Sarah Chen", "Status": "Active", "Priority": "High", "Start": "2026-04-01", "End": "2026-08-30", "Budget": 280000, "Spent": 62000, "Progress": 25},
        {"Name": "Mobile App v2", "Client": "Internal", "Manager": "Nina Patel", "Status": "Planning", "Priority": "Medium", "Start": "2026-06-01", "End": "2026-12-15", "Budget": 350000, "Spent": 0, "Progress": 0},
        {"Name": "Data Analytics Platform", "Client": "Internal", "Manager": "Alex Turner", "Status": "On Hold", "Priority": "Low", "Start": "2026-02-01", "End": "2026-06-30", "Budget": 180000, "Spent": 95000, "Progress": 55},
        {"Name": "Summit Manufacturing Portal", "Client": "Summit Industries", "Manager": "Sarah Chen", "Status": "Completed", "Priority": "High", "Start": "2025-10-01", "End": "2026-03-31", "Budget": 200000, "Spent": 192000, "Progress": 100},
    ])

def get_tasks():
    return pd.DataFrame([
        {"Task": "Database schema design", "Project": "ERP System Implementation", "Assignee": "John Park", "Status": "Done", "Priority": "High", "Due": "2026-02-15", "Est. Hours": 40, "Logged": 38},
        {"Task": "HR module development", "Project": "ERP System Implementation", "Assignee": "Rachel Lee", "Status": "Done", "Priority": "High", "Due": "2026-04-01", "Est. Hours": 120, "Logged": 115},
        {"Task": "Inventory module development", "Project": "ERP System Implementation", "Assignee": "Daniel Johnson", "Status": "In Progress", "Priority": "High", "Due": "2026-06-15", "Est. Hours": 160, "Logged": 72},
        {"Task": "Accounting integration", "Project": "ERP System Implementation", "Assignee": "John Park", "Status": "To Do", "Priority": "Medium", "Due": "2026-07-30", "Est. Hours": 80, "Logged": 0},
        {"Task": "UI/UX wireframes", "Project": "Website Redesign", "Assignee": "Sophia Nguyen", "Status": "Done", "Priority": "High", "Due": "2026-03-20", "Est. Hours": 60, "Logged": 55},
        {"Task": "Frontend development", "Project": "Website Redesign", "Assignee": "Rachel Lee", "Status": "In Progress", "Priority": "High", "Due": "2026-06-01", "Est. Hours": 200, "Logged": 140},
        {"Task": "Content migration", "Project": "Website Redesign", "Assignee": "Daniel Johnson", "Status": "Review", "Priority": "Medium", "Due": "2026-06-15", "Est. Hours": 40, "Logged": 35},
        {"Task": "API specification", "Project": "Acme Corp Integration", "Assignee": "Brian Harris", "Status": "Done", "Priority": "High", "Due": "2026-04-15", "Est. Hours": 30, "Logged": 28},
        {"Task": "Authentication setup", "Project": "Acme Corp Integration", "Assignee": "Brian Harris", "Status": "In Progress", "Priority": "High", "Due": "2026-05-30", "Est. Hours": 50, "Logged": 22},
        {"Task": "Data sync engine", "Project": "Acme Corp Integration", "Assignee": "John Park", "Status": "To Do", "Priority": "Critical", "Due": "2026-06-30", "Est. Hours": 120, "Logged": 0},
    ])

def get_revenue_data():
    return pd.DataFrame([
        {"Month": "Jan", "Revenue": 380000, "Expenses": 295000, "Profit": 85000},
        {"Month": "Feb", "Revenue": 420000, "Expenses": 310000, "Profit": 110000},
        {"Month": "Mar", "Revenue": 395000, "Expenses": 305000, "Profit": 90000},
        {"Month": "Apr", "Revenue": 510000, "Expenses": 340000, "Profit": 170000},
        {"Month": "May", "Revenue": 485000, "Expenses": 325000, "Profit": 160000},
    ])


def fmt(val):
    return f"${val:,.0f}"


# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Enterprise ERP")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Human Resources",
            "Inventory",
            "Sales",
            "Procurement",
            "Accounting",
            "Projects",
            "Reports",
            "Settings",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"v2.4.1  |  {datetime.now().strftime('%b %d, %Y')}")


# ──────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────
if page == "Dashboard":
    st.title("Dashboard")
    st.caption("Welcome back. Here's what's happening across your organization.")

    rev_data = get_revenue_data()
    products_df = get_products()
    orders_df = get_sales_orders()

    total_revenue = rev_data["Revenue"].sum()
    total_expenses = rev_data["Expenses"].sum()
    open_orders = orders_df[~orders_df["Status"].isin(["Delivered", "Cancelled"])].shape[0]
    low_stock = products_df[products_df["Status"].isin(["Low Stock", "Out of Stock"])].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue (YTD)", fmt(total_revenue), "+12.5%")
    c2.metric("Total Employees", "153", "+8")
    c3.metric("Open Orders", str(open_orders), "-3")
    c4.metric("Low Stock Alerts", str(low_stock), "+2")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Revenue vs Expenses")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rev_data["Month"], y=rev_data["Revenue"], fill="tozeroy", name="Revenue", line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=rev_data["Month"], y=rev_data["Expenses"], fill="tozeroy", name="Expenses", line=dict(color="#dc2626")))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sales by Category")
        sales_cat = pd.DataFrame({
            "Category": ["Widgets", "Gadgets", "Components", "Assemblies", "Tools", "Safety", "Raw Materials"],
            "Revenue": [285000, 420000, 195000, 310000, 145000, 88000, 165000],
        })
        fig2 = px.pie(sales_cat, values="Revenue", names="Category", hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), showlegend=True, legend=dict(font=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        st.subheader("Department Headcount")
        dept_hc = pd.DataFrame({
            "Department": ["Engineering", "Sales", "Operations", "Support", "Marketing", "Finance", "Product", "HR"],
            "Count": [42, 28, 20, 18, 15, 12, 10, 8],
        })
        fig3 = px.bar(dept_hc, x="Count", y="Department", orientation="h", color_discrete_sequence=["#2563eb"])
        fig3.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Quick Stats")
        st.info(f"**Net Profit (YTD):** {fmt(total_revenue - total_expenses)}")
        st.warning("**Overdue Invoices:** 1")
        st.success("**Pending POs:** 3")
        st.info("**Active Projects:** 3")

    with col5:
        st.subheader("Recent Activity")
        activities = [
            "Sales order SO-2026-093 created",
            "Purchase order PO-2026-046 drafted",
            "Employee leave request approved",
            "Invoice INV-2026-0324 sent",
            "Stock transfer TF-2026-012 completed",
            "Journal entry JE-2026-0158 created",
            "New deal: Pinnacle New Line",
            "Project milestone: HR Module done",
        ]
        for a in activities:
            st.markdown(f"- {a}")


# ──────────────────────────────────────────────
# HUMAN RESOURCES
# ──────────────────────────────────────────────
elif page == "Human Resources":
    st.title("Human Resources")
    st.caption("Manage employees, departments, payroll, and leave requests.")

    employees_df = get_employees()

    tab1, tab2, tab3 = st.tabs(["Employees", "Departments", "Payroll"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        active = employees_df[employees_df["Status"] == "Active"].shape[0]
        on_leave = employees_df[employees_df["Status"] == "On Leave"].shape[0]
        avg_salary = employees_df["Salary"].mean()
        c1.metric("Total Employees", str(len(employees_df)))
        c2.metric("Active", str(active))
        c3.metric("On Leave", str(on_leave))
        c4.metric("Avg. Salary", fmt(avg_salary))

        col1, col2 = st.columns([2, 1])
        with col1:
            dept_counts = employees_df["Department"].value_counts().reset_index()
            dept_counts.columns = ["Department", "Count"]
            fig = px.bar(dept_counts, x="Department", y="Count", color_discrete_sequence=["#2563eb"])
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            status_counts = employees_df["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig2 = px.pie(status_counts, values="Count", names="Status", hole=0.4, color_discrete_sequence=["#059669", "#d97706", "#dc2626"])
            fig2.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        search = st.text_input("Search employees", placeholder="Search by name, department, position...")
        filtered = employees_df
        if search:
            mask = employees_df.apply(lambda r: search.lower() in r.to_string().lower(), axis=1)
            filtered = employees_df[mask]

        st.dataframe(
            filtered[["ID", "Name", "Department", "Position", "Status", "Hire Date", "Salary", "Email"]],
            use_container_width=True,
            hide_index=True,
            column_config={"Salary": st.column_config.NumberColumn(format="$%d")},
        )

    with tab2:
        departments = pd.DataFrame([
            {"Department": "Engineering", "Head": "Sarah Chen", "Employees": 42, "Budget": 2800000},
            {"Department": "Sales", "Head": "James Miller", "Employees": 28, "Budget": 1500000},
            {"Department": "Marketing", "Head": "Emily Davis", "Employees": 15, "Budget": 900000},
            {"Department": "Human Resources", "Head": "Maria Garcia", "Employees": 8, "Budget": 450000},
            {"Department": "Finance", "Head": "Robert Kim", "Employees": 12, "Budget": 650000},
            {"Department": "Operations", "Head": "David Wilson", "Employees": 20, "Budget": 1100000},
            {"Department": "Customer Support", "Head": "Lisa Brown", "Employees": 18, "Budget": 750000},
            {"Department": "Product", "Head": "Alex Turner", "Employees": 10, "Budget": 600000},
        ])
        st.dataframe(departments, use_container_width=True, hide_index=True,
                      column_config={"Budget": st.column_config.NumberColumn(format="$%d")})

    with tab3:
        payroll = pd.DataFrame([
            {"Employee": "Sarah Chen", "Department": "Engineering", "Period": "2026-04", "Base": 15417, "Bonus": 2000, "Deductions": 3850, "Net Pay": 13567, "Status": "Processed"},
            {"Employee": "James Miller", "Department": "Sales", "Period": "2026-04", "Base": 14167, "Bonus": 5200, "Deductions": 3540, "Net Pay": 15827, "Status": "Processed"},
            {"Employee": "Robert Kim", "Department": "Finance", "Period": "2026-04", "Base": 16250, "Bonus": 1500, "Deductions": 4100, "Net Pay": 13650, "Status": "Processed"},
            {"Employee": "John Park", "Department": "Engineering", "Period": "2026-05", "Base": 11250, "Bonus": 800, "Deductions": 2800, "Net Pay": 9250, "Status": "Pending"},
            {"Employee": "Rachel Lee", "Department": "Engineering", "Period": "2026-05", "Base": 10417, "Bonus": 600, "Deductions": 2600, "Net Pay": 8417, "Status": "Pending"},
        ])
        st.dataframe(payroll, use_container_width=True, hide_index=True,
                      column_config={
                          "Base": st.column_config.NumberColumn(format="$%d"),
                          "Bonus": st.column_config.NumberColumn(format="$%d"),
                          "Deductions": st.column_config.NumberColumn(format="$%d"),
                          "Net Pay": st.column_config.NumberColumn(format="$%d"),
                      })


# ──────────────────────────────────────────────
# INVENTORY
# ──────────────────────────────────────────────
elif page == "Inventory":
    st.title("Inventory Management")
    st.caption("Track products, warehouses, and stock movements.")

    products_df = get_products()

    total_value = (products_df["Price"] * products_df["Stock"]).sum()
    total_items = products_df["Stock"].sum()
    low = products_df[products_df["Status"] == "Low Stock"].shape[0]
    oos = products_df[products_df["Status"] == "Out of Stock"].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inventory Value", fmt(total_value))
    c2.metric("Total Items", f"{total_items:,}")
    c3.metric("Low Stock", str(low))
    c4.metric("Out of Stock", str(oos))

    tab1, tab2, tab3 = st.tabs(["Products", "Warehouses", "Stock Movements"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            cat_stock = products_df.groupby("Category")["Stock"].sum().reset_index()
            fig = px.bar(cat_stock, x="Category", y="Stock", color_discrete_sequence=["#2563eb"])
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            status_counts = products_df["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig2 = px.pie(status_counts, values="Count", names="Status", hole=0.4, color_discrete_sequence=["#059669", "#d97706", "#dc2626"])
            fig2.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        search = st.text_input("Search products", placeholder="Search by name, SKU, category...")
        filtered = products_df
        if search:
            mask = products_df.apply(lambda r: search.lower() in r.to_string().lower(), axis=1)
            filtered = products_df[mask]
        st.dataframe(filtered, use_container_width=True, hide_index=True,
                      column_config={
                          "Price": st.column_config.NumberColumn(format="$%.2f"),
                          "Cost": st.column_config.NumberColumn(format="$%.2f"),
                      })

    with tab2:
        warehouses = pd.DataFrame([
            {"Name": "Main Warehouse", "Location": "Houston, TX", "Capacity": 50000, "Utilization": 78, "Manager": "Chris Anderson", "Products": 6},
            {"Name": "East Distribution", "Location": "Atlanta, GA", "Capacity": 30000, "Utilization": 62, "Manager": "Tom Richards", "Products": 3},
            {"Name": "West Facility", "Location": "Phoenix, AZ", "Capacity": 35000, "Utilization": 85, "Manager": "Susan Clark", "Products": 3},
        ])
        for _, w in warehouses.iterrows():
            with st.container():
                cc1, cc2, cc3 = st.columns([2, 1, 1])
                cc1.markdown(f"**{w['Name']}** - {w['Location']}")
                cc2.markdown(f"Manager: {w['Manager']}")
                cc3.markdown(f"Products: {w['Products']}")
                st.progress(w["Utilization"] / 100, text=f"Utilization: {w['Utilization']}% of {w['Capacity']:,} units")
                st.markdown("---")

    with tab3:
        movements = pd.DataFrame([
            {"Date": "2026-05-10", "Product": "Industrial Widget A", "Type": "IN", "Qty": 500, "From": "-", "To": "Main Warehouse", "Reference": "PO-2026-042"},
            {"Date": "2026-05-12", "Product": "Gadget Pro X1", "Type": "OUT", "Qty": 120, "From": "East Distribution", "To": "-", "Reference": "SO-2026-089"},
            {"Date": "2026-05-12", "Product": "Component Alpha", "Type": "IN", "Qty": 3000, "From": "-", "To": "Main Warehouse", "Reference": "PO-2026-045"},
            {"Date": "2026-05-13", "Product": "Premium Widget B", "Type": "TRANSFER", "Qty": 200, "From": "West Facility", "To": "Main Warehouse", "Reference": "TF-2026-012"},
            {"Date": "2026-05-14", "Product": "Assembly Kit Pro", "Type": "OUT", "Qty": 15, "From": "East Distribution", "To": "-", "Reference": "SO-2026-091"},
            {"Date": "2026-05-14", "Product": "Gadget Mini S2", "Type": "ADJUSTMENT", "Qty": -8, "From": "-", "To": "Main Warehouse", "Reference": "ADJ-2026-003"},
        ])
        st.dataframe(movements, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# SALES
# ──────────────────────────────────────────────
elif page == "Sales":
    st.title("Sales")
    st.caption("Manage sales orders, customers, and your pipeline.")

    orders_df = get_sales_orders()
    customers_df = get_customers()
    pipeline_df = get_pipeline()

    total_sales = orders_df["Total"].sum()
    avg_order = total_sales / len(orders_df)
    pipe_value = pipeline_df[~pipeline_df["Stage"].isin(["Closed Won", "Closed Lost"])]["Value"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", fmt(total_sales), "+15.3%")
    c2.metric("Customers", str(len(customers_df)), "+2")
    c3.metric("Avg. Order Value", fmt(avg_order))
    c4.metric("Pipeline Value", fmt(pipe_value), "+8.2%")

    tab1, tab2, tab3 = st.tabs(["Sales Orders", "Customers", "Pipeline"])

    with tab1:
        st.dataframe(orders_df, use_container_width=True, hide_index=True,
                      column_config={"Total": st.column_config.NumberColumn(format="$%,.2f")})

    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            seg = customers_df["Segment"].value_counts().reset_index()
            seg.columns = ["Segment", "Count"]
            fig = px.pie(seg, values="Count", names="Segment", hole=0.4, color_discrete_sequence=["#2563eb", "#d97706", "#059669"])
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(customers_df[["Name", "Company", "Segment", "Total Orders", "Total Spent", "Status"]], use_container_width=True, hide_index=True,
                          column_config={"Total Spent": st.column_config.NumberColumn(format="$%,.0f")})

    with tab3:
        stage_order = ["Lead", "Qualified", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
        pipeline_df["Stage"] = pd.Categorical(pipeline_df["Stage"], categories=stage_order, ordered=True)
        stage_data = pipeline_df.groupby("Stage", observed=False)["Value"].sum().reset_index()
        fig = px.bar(stage_data, x="Stage", y="Value", color_discrete_sequence=["#7c3aed"])
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pipeline_df.sort_values("Stage"), use_container_width=True, hide_index=True,
                      column_config={
                          "Value": st.column_config.NumberColumn(format="$%,.0f"),
                          "Probability": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                      })


# ──────────────────────────────────────────────
# PROCUREMENT
# ──────────────────────────────────────────────
elif page == "Procurement":
    st.title("Procurement")
    st.caption("Manage suppliers and purchase orders.")

    suppliers_df = get_suppliers()
    po_df = get_purchase_orders()

    total_spend = po_df["Total"].sum()
    active_suppliers = suppliers_df[suppliers_df["Status"] == "Active"].shape[0]
    pending = po_df[po_df["Status"].isin(["Draft", "Sent", "Confirmed"])].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total PO Spend", fmt(total_spend))
    c2.metric("Active Suppliers", str(active_suppliers))
    c3.metric("Pending POs", str(pending))
    c4.metric("Avg Lead Time", "10 days")

    tab1, tab2 = st.tabs(["Purchase Orders", "Suppliers"])

    with tab1:
        po_status = po_df["Status"].value_counts().reset_index()
        po_status.columns = ["Status", "Count"]
        fig = px.bar(po_status, x="Status", y="Count", color_discrete_sequence=["#2563eb"])
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(po_df, use_container_width=True, hide_index=True,
                      column_config={"Total": st.column_config.NumberColumn(format="$%,.0f")})

    with tab2:
        st.dataframe(suppliers_df, use_container_width=True, hide_index=True,
                      column_config={"Rating": st.column_config.NumberColumn(format="%.1f")})


# ──────────────────────────────────────────────
# ACCOUNTING
# ──────────────────────────────────────────────
elif page == "Accounting":
    st.title("Accounting")
    st.caption("Financial management, chart of accounts, and invoicing.")

    accounts_df = get_accounts()
    invoices_df = get_invoices()
    rev_data = get_revenue_data()

    total_assets = accounts_df[accounts_df["Type"] == "Asset"]["Balance"].sum()
    total_liabilities = accounts_df[accounts_df["Type"] == "Liability"]["Balance"].sum()
    total_revenue = accounts_df[accounts_df["Type"] == "Revenue"]["Balance"].sum()
    total_expenses = accounts_df[accounts_df["Type"] == "Expense"]["Balance"].sum()
    ar = accounts_df[accounts_df["Code"] == "1100"]["Balance"].values[0]
    ap = accounts_df[accounts_df["Code"] == "2000"]["Balance"].values[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Assets", fmt(total_assets))
    c2.metric("Accounts Receivable", fmt(ar))
    c3.metric("Accounts Payable", fmt(ap))
    c4.metric("Net Income (YTD)", fmt(total_revenue - total_expenses), "+18.4%")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Chart of Accounts", "Invoices", "Financial Statements"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            type_balances = accounts_df.groupby("Type")["Balance"].sum().abs().reset_index()
            fig = px.bar(type_balances, x="Type", y="Balance", color="Type",
                         color_discrete_map={"Asset": "#2563eb", "Liability": "#dc2626", "Equity": "#059669", "Revenue": "#7c3aed", "Expense": "#d97706"})
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            inv_status = invoices_df["Status"].value_counts().reset_index()
            inv_status.columns = ["Status", "Count"]
            fig2 = px.pie(inv_status, values="Count", names="Status", hole=0.4,
                          color_discrete_map={"Paid": "#059669", "Sent": "#2563eb", "Overdue": "#dc2626"})
            fig2.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.dataframe(accounts_df, use_container_width=True, hide_index=True,
                      column_config={"Balance": st.column_config.NumberColumn(format="$%,.0f")})

    with tab3:
        st.dataframe(invoices_df, use_container_width=True, hide_index=True,
                      column_config={
                          "Total": st.column_config.NumberColumn(format="$%,.2f"),
                          "Paid": st.column_config.NumberColumn(format="$%,.2f"),
                          "Balance": st.column_config.NumberColumn(format="$%,.2f"),
                      })

    with tab4:
        st.subheader("Income Statement (YTD)")
        cogs = accounts_df[accounts_df["Code"] == "5000"]["Balance"].values[0]
        gross_profit = total_revenue - cogs
        opex = total_expenses - cogs
        net_income = gross_profit - opex

        is_data = pd.DataFrame([
            {"Item": "Total Revenue", "Amount": total_revenue},
            {"Item": "Cost of Goods Sold", "Amount": -cogs},
            {"Item": "Gross Profit", "Amount": gross_profit},
            {"Item": "Operating Expenses", "Amount": -opex},
            {"Item": "Net Income", "Amount": net_income},
        ])
        st.dataframe(is_data, use_container_width=True, hide_index=True,
                      column_config={"Amount": st.column_config.NumberColumn(format="$%,.0f")})

        st.markdown("---")
        st.subheader("Balance Sheet Summary")
        bs_data = pd.DataFrame([
            {"Item": "Total Assets", "Amount": total_assets},
            {"Item": "Total Liabilities", "Amount": total_liabilities},
            {"Item": "Total Equity", "Amount": total_assets - total_liabilities},
        ])
        st.dataframe(bs_data, use_container_width=True, hide_index=True,
                      column_config={"Amount": st.column_config.NumberColumn(format="$%,.0f")})

        col1, col2, col3 = st.columns(3)
        col1.metric("Gross Margin", f"{(gross_profit / total_revenue * 100):.1f}%")
        col2.metric("Net Margin", f"{(net_income / total_revenue * 100):.1f}%")
        col3.metric("Current Ratio", f"{(total_assets / total_liabilities):.2f}")


# ──────────────────────────────────────────────
# PROJECTS
# ──────────────────────────────────────────────
elif page == "Projects":
    st.title("Projects")
    st.caption("Manage projects, tasks, and track progress.")

    projects_df = get_projects()
    tasks_df = get_tasks()

    active = projects_df[projects_df["Status"] == "Active"].shape[0]
    total_budget = projects_df["Budget"].sum()
    total_spent = projects_df["Spent"].sum()
    total_hours = tasks_df["Logged"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Projects", str(active))
    c2.metric("Total Budget", fmt(total_budget))
    c3.metric("Total Spent", fmt(total_spent))
    c4.metric("Hours Logged", f"{total_hours}h")

    tab1, tab2, tab3 = st.tabs(["Projects", "Tasks", "Task Board"])

    with tab1:
        for _, p in projects_df.iterrows():
            with st.expander(f"{p['Name']} ({p['Status']}) - {p['Client']}"):
                pc1, pc2, pc3 = st.columns(3)
                pc1.markdown(f"**Manager:** {p['Manager']}")
                pc2.markdown(f"**Priority:** {p['Priority']}")
                pc3.markdown(f"**Timeline:** {p['Start']} to {p['End']}")
                st.progress(p["Progress"] / 100, text=f"Progress: {p['Progress']}%")
                bc1, bc2 = st.columns(2)
                bc1.metric("Budget", fmt(p["Budget"]))
                bc2.metric("Spent", fmt(p["Spent"]))

    with tab2:
        st.dataframe(tasks_df, use_container_width=True, hide_index=True,
                      column_config={
                          "Est. Hours": st.column_config.NumberColumn(format="%dh"),
                          "Logged": st.column_config.NumberColumn(format="%dh"),
                      })

    with tab3:
        cols = st.columns(4)
        statuses = ["To Do", "In Progress", "Review", "Done"]
        for i, status in enumerate(statuses):
            with cols[i]:
                st.markdown(f"### {status}")
                filtered = tasks_df[tasks_df["Status"] == status]
                for _, t in filtered.iterrows():
                    st.markdown(f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
<strong>{t['Task']}</strong><br/>
<span style="font-size: 0.8rem; color: #6b7280;">{t['Project']}</span><br/>
<span style="font-size: 0.75rem; color: #9ca3af;">{t['Assignee']} | {t['Priority']}</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# REPORTS
# ──────────────────────────────────────────────
elif page == "Reports":
    st.title("Reports & Analytics")
    st.caption("Comprehensive business intelligence across all modules.")

    rev_data = get_revenue_data()
    accounts_df = get_accounts()

    total_revenue_all = rev_data["Revenue"].sum()
    cogs = accounts_df[accounts_df["Code"] == "5000"]["Balance"].values[0]
    gross_profit = total_revenue_all - cogs
    opex = accounts_df[(accounts_df["Type"] == "Expense") & (accounts_df["Code"] != "5000")]["Balance"].sum()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue & Profit Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rev_data["Month"], y=rev_data["Revenue"], fill="tozeroy", name="Revenue", line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=rev_data["Month"], y=rev_data["Profit"], fill="tozeroy", name="Profit", line=dict(color="#059669")))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cash Flow Analysis")
        cf = pd.DataFrame({
            "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
            "Inflow": [410000, 445000, 425000, 540000, 510000],
            "Outflow": [345000, 360000, 355000, 388000, 375000],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=cf["Month"], y=cf["Inflow"], name="Inflow", marker_color="#059669"))
        fig2.add_trace(go.Bar(x=cf["Month"], y=cf["Outflow"], name="Outflow", marker_color="#dc2626"))
        fig2.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20), barmode="group", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Profit Margin Trend")
        margins = rev_data.copy()
        margins["Margin %"] = (margins["Profit"] / margins["Revenue"] * 100).round(1)
        fig3 = px.line(margins, x="Month", y="Margin %", markers=True, color_discrete_sequence=["#7c3aed"])
        fig3.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Inventory Turnover")
        turnover = pd.DataFrame({
            "Category": ["Components", "Raw Materials", "Widgets", "Gadgets", "Safety", "Assemblies", "Tools"],
            "Turnover": [6.1, 5.5, 4.2, 3.8, 3.2, 2.9, 1.5],
        })
        fig4 = px.bar(turnover, x="Turnover", y="Category", orientation="h", color_discrete_sequence=["#0891b2"])
        fig4.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Income Statement Summary")
    isc1, isc2, isc3 = st.columns(3)
    isc1.metric("Total Revenue", fmt(total_revenue_all))
    isc2.metric("Gross Profit", fmt(gross_profit))
    isc3.metric("Net Income", fmt(gross_profit - opex))

    m1, m2, m3 = st.columns(3)
    m1.metric("Gross Margin", f"{(gross_profit / total_revenue_all * 100):.1f}%")
    m2.metric("Net Margin", f"{((gross_profit - opex) / total_revenue_all * 100):.1f}%")
    total_assets = accounts_df[accounts_df["Type"] == "Asset"]["Balance"].sum()
    total_liabilities = accounts_df[accounts_df["Type"] == "Liability"]["Balance"].sum()
    m3.metric("Current Ratio", f"{(total_assets / total_liabilities):.2f}")


# ──────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────
elif page == "Settings":
    st.title("Settings")
    st.caption("Manage your organization and system configuration.")

    tab1, tab2, tab3 = st.tabs(["Company", "Users & Roles", "System"])

    with tab1:
        st.subheader("Company Information")
        col1, col2 = st.columns(2)
        col1.text_input("Company Name", value="Acme Industries Inc.")
        col2.selectbox("Industry", ["Manufacturing", "Technology", "Retail", "Services"])
        col3, col4 = st.columns(2)
        col3.text_input("Tax ID", value="XX-XXXXXXX")
        col4.selectbox("Currency", ["USD - US Dollar", "EUR - Euro", "GBP - British Pound"])
        st.text_area("Address", value="100 Enterprise Blvd, Houston, TX 77001")
        col5, col6 = st.columns(2)
        col5.text_input("Phone", value="+1-555-0100")
        col6.text_input("Website", value="https://www.acme-industries.com")

        st.markdown("---")
        st.subheader("Localization")
        lc1, lc2, lc3 = st.columns(3)
        lc1.selectbox("Language", ["English (US)", "Spanish", "French", "German", "Arabic"])
        lc2.selectbox("Timezone", ["America/Chicago (CST)", "America/New_York (EST)", "America/Los_Angeles (PST)"])
        lc3.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])

    with tab2:
        st.subheader("Users")
        users_data = pd.DataFrame([
            {"Name": "Admin User", "Email": "admin@company.com", "Role": "Super Admin", "Status": "Online"},
            {"Name": "Sarah Chen", "Email": "sarah.chen@company.com", "Role": "Manager", "Status": "Online"},
            {"Name": "Robert Kim", "Email": "robert.kim@company.com", "Role": "Finance Admin", "Status": "Away"},
            {"Name": "James Miller", "Email": "james.miller@company.com", "Role": "Sales Manager", "Status": "Offline"},
            {"Name": "Maria Garcia", "Email": "maria.garcia@company.com", "Role": "HR Manager", "Status": "Online"},
        ])
        st.dataframe(users_data, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Role Permissions")
        for role in ["Super Admin", "Manager", "Finance Admin", "Sales Manager", "HR Manager", "Viewer"]:
            st.checkbox(role, value=True, key=f"role_{role}")

    with tab3:
        st.subheader("System Information")
        info = {
            "Application Version": "2.4.1",
            "Database": "PostgreSQL 16.2",
            "Last Backup": "May 19, 2026, 3:00 AM CST",
            "Storage Used": "42.5 GB / 100 GB (42.5%)",
            "API Rate Limit": "1,000 requests/minute",
            "Uptime": "99.97% (last 30 days)",
        }
        for k, v in info.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.button("Run Backup", type="secondary")
        col2.button("Clear Cache", type="secondary")
