# Before running this script, set these environment variables:
# On macOS/Linux:
# export JIRA_USERNAME='your-jira-email@example.com'
# export JIRA_API_TOKEN='your-jira-api-token'
# export OPENAI_API_KEY='your-openai-api-key'
#
# Or on Windows:
# set JIRA_USERNAME=your-jira-email@example.com'
# set JIRA_API_TOKEN=your-jira-api-token'
# set OPENAI_API_KEY=your-openai-api-key'
#
# To get these values:
# 1. JIRA_USERNAME: Your Atlassian account email
# 2. JIRA_API_TOKEN: Generate from https://id.atlassian.com/manage-profile/security/api-tokens
# 3. OPENAI_API_KEY: Get from https://platform.openai.com/api-keys

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
import hashlib
import json
import time
import re
from dataclasses import dataclass
from functools import wraps
from dotenv import load_dotenv
import plotly.express as px

# Configure page layout
st.set_page_config(
    page_title="JIRA Ticket Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-16idsys {
        padding-top: 2rem;
    }
    .st-emotion-cache-18ni7ap {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables from .env file
load_dotenv()

# Remove spaces from OpenAI API key if present
openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()

# Initialize OpenAI client once
try:
    client = OpenAI(
        api_key=openai_api_key
    )
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    logger.error(f"OpenAI client initialization error: {str(e)}")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration management
class Config:
    def __init__(self):
        self.JIRA_URL = os.getenv('JIRA_URL')
        self.JIRA_API_ENDPOINT = f"{self.JIRA_URL}/rest/api/2/search"
        self.JIRA_USERNAME = os.getenv('JIRA_USERNAME')
        self.JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.PROJECT_KEY = os.getenv('JIRA_PROJECT_KEY')  # Make project key configurable
        
        self.validate_config()
    
    def validate_config(self):
        missing_vars = []
        for var in ['JIRA_USERNAME', 'JIRA_API_TOKEN', 'OPENAI_API_KEY']:
            if not getattr(self, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize configuration
try:
    config = Config() 
except ValueError as e:
    st.error(f"Configuration Error: {str(e)}")
    st.stop()

@lru_cache(maxsize=1000)
def get_cached_categorization(text_hash: str) -> str:
    """Cached version of OpenAI categorization to avoid redundant API calls."""
    return st.session_state.get(f'category_cache_{text_hash}', None)

def set_cached_categorization(text_hash: str, category: str) -> None:
    """Store categorization in cache."""
    st.session_state[f'category_cache_{text_hash}'] = category

def hash_text(text: str) -> str:
    """Create a hash of the text for caching purposes."""
    return hashlib.md5(text.encode()).hexdigest()

def secure_api_call(func):
    """Decorator to add security headers and request signing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add security headers
        headers = kwargs.get('headers', {})
        headers.update({
            'X-Request-Timestamp': str(int(time.time())),
            'X-Client-Version': '1.0.0',
            'User-Agent': 'JIRA-Analysis-Tool/1.0'
        })
        kwargs['headers'] = headers
        return func(*args, **kwargs)
    return wrapper

def rate_limit(calls: int, period: int):
    """Rate limiting decorator to prevent API abuse."""
    def decorator(func):
        last_reset = time.time()
        calls_made = 0
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_reset, calls_made
            current_time = time.time()
            if current_time - last_reset >= period:
                calls_made = 0
                last_reset = current_time
            if calls_made >= calls:
                wait_time = period - (current_time - last_reset)
                if wait_time > 0:
                    time.sleep(wait_time)
                    last_reset = time.time()
                    calls_made = 0
            calls_made += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@rate_limit(calls=3, period=60)
@secure_api_call
def get_openai_categorization(summary: str, description: str) -> str:
    """Get ticket categorization from OpenAI with proper error handling."""
    try:
        # Ensure inputs are strings and not None
        summary = str(summary) if summary is not None else ''
        description = str(description) if description is not None else ''
        
        text_hash = hash_text(f"{summary}{description}")
        cached_result = get_cached_categorization(text_hash)
        if cached_result:
            return cached_result

        prompt = f"""Categorize the following JIRA ticket into one of these categories: Aggregate, EPCIS, MDN, or Shipping.
        Ticket Summary: {summary}
        Ticket Description: {description}
        Return only the category name."""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Changed from gpt-4o to gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that categorizes JIRA tickets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            if not response.choices:
                logger.error("No response from OpenAI API")
                return "Unknown"
                
            category = response.choices[0].message.content.strip()
            valid_categories = ["Aggregate", "EPCIS", "MDN", "Shipping"]
            result = category if category in valid_categories else "Unknown"
            
            set_cached_categorization(text_hash, result)
            return result
            
        except AttributeError as e:
            logger.error(f"OpenAI API response format error: {str(e)}")
            return "Unknown"
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return "Unknown"
            
    except Exception as e:
        logger.error(f"Error in categorization: {str(e)}")
        return "Unknown"

def categorize_ticket(summary: str, description: str) -> Tuple[str, Dict[str, str]]:
    """Categorize a ticket using keyword-based matching."""
    try:
        # Ensure inputs are strings
        summary = str(summary) if summary is not None else ''
        description = str(description) if description is not None else ''
        
        # Keyword-based categorization
        categories = {
            "Aggregate": {"aggregate", "aggregation", "batch processing", "lot"},
            "EPCIS": {"epcis", "event capture", "tracking", "serialization", "traceability", "events"},
            "MDN": {"mdn", "message disposition", "acknowledgment", "receipt", "edi", "message"},
            "Shipping": {"shipping", "shipment", "ship", "carrier", "freight", "logistics", "transportation", "delivery"}
        }

        ticket_text = f"{summary} {description}".lower()
        scores = {category: 0 for category in categories}
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in ticket_text:
                    scores[category] += 1

        highest_score = max(scores.values())
        if highest_score > 0:
            highest_category = max(scores.items(), key=lambda x: x[1])[0]
        else:
            highest_category = "Unknown"
        
        categorized = {category: "TRUE" if category == highest_category else "FALSE"
                      for category in categories}
        
        return highest_category, categorized
        
    except Exception as e:
        logger.error(f"Error in categorization: {str(e)}")
        return "Unknown", {
            "Aggregate": "FALSE",
            "EPCIS": "FALSE",
            "MDN": "FALSE",
            "Shipping": "FALSE"
        }

def process_tickets_in_batches(tickets: List[Dict], batch_size: int = 50) -> List[Dict]:
    """Process tickets in batches with improved error handling"""
    if not tickets:
        logger.warning("No tickets to process")
        return []
        
    processed_tickets = []
    total_batches = (len(tickets) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(tickets))
        batch = tickets[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch)} tickets)")
        
        for ticket in batch:
            try:
                if not isinstance(ticket, dict):
                    logger.warning(f"Skipping invalid ticket format: {type(ticket)}")
                    continue
                
                # Ensure required fields exist
                required_fields = {'key', 'summary', 'description'}
                if not all(field in ticket for field in required_fields):
                    logger.warning(f"Skipping ticket {ticket.get('key', 'UNKNOWN')}: Missing required fields")
                    continue
                
                # Process the ticket
                summary = str(ticket.get('summary', ''))
                description = str(ticket.get('description', ''))
                
                try:
                    category, categorized = categorize_ticket(summary, description)
                    ticket['category'] = category
                    ticket['categorization'] = categorized
                    processed_tickets.append(ticket)
                except Exception as e:
                    logger.error(f"Error categorizing ticket {ticket.get('key', 'UNKNOWN')}: {str(e)}")
                    # Add ticket with default category
                    ticket['category'] = "Unknown"
                    ticket['categorization'] = {cat: "FALSE" for cat in ["Aggregate", "EPCIS", "MDN", "Shipping"]}
                    processed_tickets.append(ticket)
                    
            except Exception as e:
                logger.error(f"Error processing ticket in batch {batch_num}: {str(e)}")
                continue
                
        logger.info(f"Completed batch {batch_num + 1}/{total_batches}")
    
    return processed_tickets

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@rate_limit(calls=30, period=60)  # Limit to 30 calls per minute
def fetch_jira_tickets(jql: str) -> List[Dict]:
    """Fetch JIRA tickets with rate limiting and validation."""
    try:
        validated_jql = validate_jql(jql)
        auth = HTTPBasicAuth(config.JIRA_USERNAME, config.JIRA_API_TOKEN)
        headers = {'Accept': 'application/json'}
        ticket_data = []
        start_at = 0
        batch_size = 100
        timeout = 30

        # Test JIRA connection first
        test_response = requests.get(
            config.JIRA_API_ENDPOINT,
            headers=headers,
            params={'jql': 'project = ' + config.PROJECT_KEY + ' AND created >= startOfDay()', 'maxResults': 1},
            auth=auth,
            timeout=timeout
        )
        test_response.raise_for_status()

        while True:
            try:
                params = {
                    'jql': validated_jql,
                    'startAt': start_at,
                    'maxResults': batch_size,
                    'fields': 'summary,description,created,updated,status'  # Removed customfields
                }
                
                response = requests.get(
                    config.JIRA_API_ENDPOINT,
                    headers=headers,
                    params=params,
                    auth=auth,
                    timeout=timeout
                )
                response.raise_for_status()
                
                data = response.json()
                if not data.get('issues'):
                    break
                    
                batch_tickets = [format_ticket(issue) for issue in data['issues']]
                processed_batch = process_tickets_in_batches(batch_tickets)
                ticket_data.extend(processed_batch)
                
                if len(data['issues']) < batch_size:
                    break
                    
                start_at += batch_size
                
            except requests.exceptions.RequestException as e:
                logger.error(f"JIRA API Error: {str(e)}")
                if hasattr(e.response, 'text'):
                    logger.error(f"Response details: {e.response.text}")
                raise
                
        return ticket_data
    except Exception as e:
        logger.error(f"Error fetching JIRA tickets: {str(e)}")
        if isinstance(e, requests.exceptions.HTTPError):
            logger.error(f"HTTP Error response: {e.response.text if hasattr(e, 'response') else 'No response text'}")
        raise

def format_ticket(issue: Dict) -> Dict:
    """Format a JIRA issue into a consistent structure."""
    try:
        fields = issue.get('fields', {})
        return {
            'key': issue.get('key', 'UNKNOWN'),
            'summary': str(fields.get('summary', '')),
            'description': str(fields.get('description', '')),
            'created': fields.get('created', ''),
            'status': fields.get('status', {}).get('name', 'Unknown'),
            'resolved': None,
            'organization': str(fields.get('customfield_10024', 'Unknown')),
            'severity': str(fields.get('customfield_10002', 'Unknown'))
        }
    except Exception as e:
        logger.error(f"Error formatting ticket: {str(e)}")
        return {
            'key': 'ERROR',
            'summary': '',
            'description': '',
            'created': '',
            'status': 'Unknown',
            'resolved': None,
            'organization': 'Unknown',
            'severity': 'Unknown'
        }

def make_timezone_aware(date_input):
    if isinstance(date_input, datetime):
        return date_input.astimezone(timezone.utc)
    try:
        dt = datetime.strptime(date_input, '%Y-%m-%dT%H:%M:%S.%f%z')
    except ValueError:
        dt = datetime.strptime(date_input, '%Y-%m-%dT%H:%M:%S%z')
    return dt.astimezone(timezone.utc)

def display_high_level_info(open_tickets, closed_tickets, start_date, end_date):
    try:
        # Filter open tickets based on creation date
        filtered_open_tickets = [ticket for ticket in open_tickets if
                             start_date <= make_timezone_aware(ticket['created']) <= end_date]
        
        # Filter closed tickets based on creation date since 'updated' might not be available
        filtered_closed_tickets = [ticket for ticket in closed_tickets if
                               start_date <= make_timezone_aware(ticket['created']) <= end_date]

        new_tickets = len(filtered_open_tickets)
        closed_tickets_count = len(filtered_closed_tickets)
        pending_tickets = len(filtered_open_tickets)

        st.write("### Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="New Tickets", value=new_tickets)
            
        with col2:
            st.metric(label="Closed Tickets", value=closed_tickets_count)
            
        with col3:
            st.metric(label="Pending Tickets", value=pending_tickets)

    except Exception as e:
        logger.error(f"Error in display_high_level_info: {str(e)}")
        st.error("Error displaying summary information. Please check the logs for details.")

def display_tickets(ticket_data, start_date, end_date, selected_status):
    st.write("### Ticket Details")
    
    try:
        filtered_tickets = [
            ticket for ticket in ticket_data
            if start_date <= make_timezone_aware(ticket['created']) <= end_date
               and (ticket['status'] == selected_status or selected_status == "All")
        ]
        
        if not filtered_tickets:
            st.info("No tickets found for the selected criteria")
            return
            
        df = pd.DataFrame([{
            "Ticket": ticket['key'],
            "Title": ticket['summary'],
            "Status": ticket['status'],
            "Created": pd.to_datetime(ticket['created']).strftime('%Y-%m-%d'),
            "Category": ticket.get('category', 'Uncategorized')
        } for ticket in filtered_tickets])
        
        # Add search functionality
        search_term = st.text_input("Search tickets", "")
        if search_term:
            df = df[df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
        
        # Add sorting functionality
        sort_column = st.selectbox("Sort by", df.columns)
        sort_order = st.radio("Sort order", ["Ascending", "Descending"], horizontal=True)
        df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
        
        # Display the table with improved styling
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticket": st.column_config.TextColumn(width="medium"),
                "Title": st.column_config.TextColumn(width="large"),
                "Status": st.column_config.TextColumn(width="small"),
                "Created": st.column_config.TextColumn(width="small"),
                "Category": st.column_config.TextColumn(width="medium")
            }
        )
        
    except Exception as e:
        st.error(f"Error displaying tickets: {str(e)}")
        logging.error(f"Ticket display error: {str(e)}")

def display_charts(open_tickets, closed_tickets, start_date, end_date):
    try:
        st.write("## Ticket Trends Analysis")
        col1, col2 = st.columns(2)
        
        # Filter tickets based on creation date only
        filtered_open_tickets = [ticket for ticket in open_tickets if
                                start_date <= make_timezone_aware(ticket['created']) <= end_date]
        filtered_closed_tickets = [ticket for ticket in closed_tickets if
                                start_date <= make_timezone_aware(ticket['created']) <= end_date]

        if not filtered_open_tickets and not filtered_closed_tickets:
            st.warning("No data available for the selected date range")
            return

        df_open = pd.DataFrame(filtered_open_tickets)
        df_closed = pd.DataFrame(filtered_closed_tickets)
        
        # Ticket Volume Trend
        with col1:
            st.write("### Ticket Volume Trend")
            if not df_open.empty:
                df_open['created_date'] = pd.to_datetime(df_open['created']).dt.date
                created_counts = df_open.groupby('created_date').size().reset_index(name='count')
                
                fig = px.line(created_counts, x='created_date', y='count',
                            title='Daily Ticket Creation Trend',
                            labels={'created_date': 'Date', 'count': 'Number of Tickets'})
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    hovermode='x unified',
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)

        # Status Distribution
        with col2:
            st.write("### Status Distribution")
            all_tickets_df = pd.concat([df_open, df_closed]) if not df_closed.empty else df_open
            if not all_tickets_df.empty:
                status_counts = all_tickets_df['status'].value_counts()
                fig = px.pie(values=status_counts.values, 
                           names=status_counts.index,
                           title='Ticket Status Distribution')
                fig.update_layout(
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        # Category Analysis
        st.write("### Category Analysis")
        if not all_tickets_df.empty and 'category' in all_tickets_df.columns:
            category_counts = all_tickets_df['category'].value_counts()
            fig = px.bar(x=category_counts.index, 
                        y=category_counts.values,
                        title='Ticket Categories',
                        labels={'x': 'Category', 'y': 'Count'})
            fig.update_layout(
                height=400,
                xaxis_tickangle=-45,
                plot_bgcolor='white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying charts: {str(e)}")
        logging.error(f"Chart display error: {str(e)}")

def visualize_categorized_tickets(open_tickets, closed_tickets, start_date, end_date):
    st.write("## Ticket Categories Analysis")
    
    try:
        # Combine and filter tickets
        all_tickets = open_tickets + closed_tickets
        filtered_tickets = [
            ticket for ticket in all_tickets
            if start_date <= make_timezone_aware(ticket['created']) <= end_date
        ]
        
        if not filtered_tickets:
            st.info("No tickets found for the selected date range")
            return
            
        # Create columns for different visualizations
        col1, col2 = st.columns(2)
        
        # Category by Status
        with col1:
            df = pd.DataFrame(filtered_tickets)
            if 'category' in df.columns and 'status' in df.columns:
                pivot_table = pd.crosstab(df['category'], df['status'])
                fig = px.bar(pivot_table, 
                            title='Categories by Status',
                            barmode='group',
                            labels={'value': 'Count', 'status': 'Status'})
                fig.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    plot_bgcolor='white',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Category Trend Over Time
        with col2:
            df['created_date'] = pd.to_datetime(df['created']).dt.date
            category_trend = df.groupby(['created_date', 'category']).size().reset_index(name='count')
            fig = px.line(category_trend, 
                         x='created_date',
                         y='count',
                         color='category',
                         title='Category Trends Over Time')
            fig.update_layout(
                height=400,
                xaxis_title='Date',
                yaxis_title='Number of Tickets',
                plot_bgcolor='white',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error visualizing categorized tickets: {str(e)}")
        logging.error(f"Category visualization error: {str(e)}")

def query_llm(question, context, open_ticket_statuses, closed_ticket_statuses):
    """Query OpenAI with proper error handling"""
    try:
        limited_context = "\n\n".join(context.split("\n\n")[:100000])

        open_statuses_str = ", ".join(open_ticket_statuses)
        closed_statuses_str = ", ".join(closed_ticket_statuses)

        prompt = (
            "You are an AI assistant specialized in analyzing JIRA tickets. Based on the following ticket descriptions and categories, "
            "please provide a consistent analysis and answer the user's question in a precise and unambiguous manner.\n\n"
            f"Context:\n{limited_context}\n\n"
            f"Open Ticket Statuses: {open_statuses_str}\n"
            f"Closed Ticket Statuses: {closed_statuses_str}\n\n"
            f"Question:\n{question}\n\n"
            "Please provide a consistent and accurate response:"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides insights based on JIRA ticket descriptions & summary."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in LLM query: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    """Validate the date range for JIRA queries."""
    if start_date > end_date:
        raise ValueError("Start date must be before end date")
    if end_date > datetime.now(timezone.utc):
        raise ValueError("End date cannot be in the future")
    if (end_date - start_date).days > 365:
        raise ValueError("Date range cannot exceed one year")

def sanitize_jql(jql: str) -> str:
    """Basic JQL injection prevention."""
    # Remove any attempts at JQL injection
    dangerous_chars = [';', '"', "'"]
    for char in dangerous_chars:
        jql = jql.replace(char, '')
    return jql

def validate_jql(jql: str) -> str:
    """Validate and sanitize JQL query."""
    # Allow single quotes in JQL as they are valid
    if ';' in jql:
        raise ValueError("Invalid characters in JQL query")
    
    # Ensure basic JQL structure
    if not any(keyword in jql.lower() for keyword in ['project', 'created', 'updated']):
        raise ValueError("JQL query must contain at least a project, created, or updated specification")
        
    return jql

def process_jira_data(tickets: List[Dict]) -> pd.DataFrame:
    """Process JIRA ticket data into a pandas DataFrame with error handling."""
    try:
        df = pd.DataFrame(tickets)
        
        # Convert date columns to datetime
        for date_col in ['created', 'updated', 'resolved']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], utc=True)
        
        # Fill missing values
        df['description'] = df['description'].fillna('No description provided')
        df['organization'] = df['organization'].fillna('Unknown')
        df['severity'] = df['severity'].fillna('Unknown')
        
        return df
    except Exception as e:
        logger.error(f"Error processing JIRA data: {str(e)}")
        return pd.DataFrame()

@dataclass
class JiraTicket:
    """Data class for validating JIRA ticket structure."""
    key: str
    summary: str
    description: str
    created: str
    status: str
    resolved: Optional[str]
    organization: str
    severity: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JiraTicket':
        """Validate and create a JiraTicket from dictionary data."""
        required_fields = {'key', 'summary', 'created', 'status'}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        return cls(
            key=str(data['key']),
            summary=str(data['summary']),
            description=str(data.get('description', '')),
            created=str(data['created']),
            status=str(data['status']),
            resolved=str(data['resolved']) if data.get('resolved') else None,
            organization=str(data.get('organization', 'Unknown')),
            severity=str(data.get('severity', 'Unknown'))
        )

def init_session_state():
    """Initialize session state variables."""
    if 'category_cache' not in st.session_state:
        st.session_state.category_cache = {}
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = time.time()
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

def main():
    """Main application entry point with proper error handling."""
    try:
        # Initialize session state
        init_session_state()
        
        # Validate user session
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = time.time()
        elif time.time() - st.session_state.last_activity > 3600:  # 1 hour timeout
            st.session_state.clear()
            st.error("Session expired. Please refresh the page.")
            return
            
        # Update last activity
        st.session_state.last_activity = time.time()
        
        st.title("JIRA Ticket Analysis Dashboard")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                     datetime.now(timezone.utc) - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", 
                                   datetime.now(timezone.utc))
        
        start_date = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_date = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
        
        # Fetch and process data
        jql = sanitize_jql(f"created >= '{start_date.date()}' AND created <= '{end_date.date()}'")
        tickets = fetch_jira_tickets(jql)
        
        if not tickets:
            st.warning("No tickets found for the selected date range.")
            return
            
        df = process_jira_data(tickets)
        
        # Display visualizations
        display_high_level_info(
            df[df['status'] != 'Closed'].to_dict('records'),
            df[df['status'] == 'Closed'].to_dict('records'),
            start_date,
            end_date
        )
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()

# Streamlit UI
st.sidebar.title("JIRA Ticket Monitor")
days_selected = st.sidebar.selectbox("Select Number of Days", list(range(1, 366)))
#selected_date = st.sidebar.date_input("Select Date", datetime.now(timezone.utc).date())
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=days_selected)

if isinstance(start_date, datetime):
    start_date = start_date.replace(tzinfo=timezone.utc)
if isinstance(end_date, datetime):
    end_date = end_date.replace(tzinfo=timezone.utc)

open_ticket_statuses = [
    "To Do",
    "In Progress",
    "QA"
]
closed_ticket_statuses = [
    "Done"
]
open_ticket_statuses_str = '","'.join(open_ticket_statuses)
open_tickets_jql = f'project = "{config.PROJECT_KEY}" AND status in ("{open_ticket_statuses_str}") AND created >= "{start_date.strftime("%Y-%m-%d")}" AND created <= "{end_date.strftime("%Y-%m-%d")}" ORDER BY created DESC'
closed_ticket_statuses_str = '","'.join(closed_ticket_statuses)
closed_tickets_jql = f'project = "{config.PROJECT_KEY}" AND status in ("{closed_ticket_statuses_str}") AND created >= "{start_date.strftime("%Y-%m-%d")}" AND created <= "{end_date.strftime("%Y-%m-%d")}" ORDER BY created DESC'

open_tickets = fetch_jira_tickets(open_tickets_jql)
closed_tickets = fetch_jira_tickets(closed_tickets_jql)

status_options = ["All"] + sorted(set(ticket['status'] for ticket in open_tickets + closed_tickets))
selected_status = st.sidebar.selectbox("Select Status", status_options)
if st.sidebar.button('Fetch Data'):
    display_high_level_info(open_tickets, closed_tickets, start_date, end_date)
    #display_tickets(open_tickets, start_date, end_date, selected_status)
    display_charts(open_tickets, closed_tickets, start_date, end_date)
    visualize_categorized_tickets(open_tickets, closed_tickets, start_date, end_date)

def analyze_tickets_basic(query: str, tickets: List[Dict], start_date: datetime, end_date: datetime) -> str:
    """Basic ticket analysis without using OpenAI."""
    try:
        filtered_tickets = [t for t in tickets if start_date <= make_timezone_aware(t['created']) <= end_date]
        
        if 'status' in query.lower():
            status_counts = {}
            for ticket in filtered_tickets:
                status = ticket['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            return f"Status distribution: {json.dumps(status_counts, indent=2)}"
            
        if 'category' in query.lower():
            category_counts = {}
            for ticket in filtered_tickets:
                category = ticket.get('category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            return f"Category distribution: {json.dumps(category_counts, indent=2)}"
            
        if 'count' in query.lower():
            return f"Total tickets in selected period: {len(filtered_tickets)}"
            
        return "Please ask about status distribution, category distribution, or ticket count."
        
    except Exception as e:
        logger.error(f"Error in basic analysis: {str(e)}")
        return f"Error analyzing tickets: {str(e)}"

def analyze_tickets_with_ai(query: str, tickets: List[Dict], start_date: datetime, end_date: datetime) -> str:
    """Analyze tickets using OpenAI for natural language understanding."""
    try:
        filtered_tickets = [t for t in tickets if start_date <= make_timezone_aware(t['created']) <= end_date]
        
        # Prepare context for OpenAI
        ticket_summary = {
            'total_tickets': len(filtered_tickets),
            'date_range': f"{start_date.date()} to {end_date.date()}",
            'status_distribution': {},
            'category_distribution': {},
            'daily_counts': {}
        }
        
        # Collect statistics
        for ticket in filtered_tickets:
            # Status distribution
            status = ticket['status']
            ticket_summary['status_distribution'][status] = ticket_summary['status_distribution'].get(status, 0) + 1
            
            # Category distribution
            category = ticket.get('category', 'Unknown')
            ticket_summary['category_distribution'][category] = ticket_summary['category_distribution'].get(category, 0) + 1
            
            # Daily counts
            created_date = make_timezone_aware(ticket['created']).date().isoformat()
            ticket_summary['daily_counts'][created_date] = ticket_summary['daily_counts'].get(created_date, 0) + 1

        context = f"""
        Ticket Analysis Context:
        Time Period: {ticket_summary['date_range']}
        Total Tickets: {ticket_summary['total_tickets']}
        
        Status Distribution:
        {json.dumps(ticket_summary['status_distribution'], indent=2)}
        
        Category Distribution:
        {json.dumps(ticket_summary['category_distribution'], indent=2)}
        
        Daily Ticket Counts:
        {json.dumps(dict(sorted(ticket_summary['daily_counts'].items())), indent=2)}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a JIRA ticket analysis expert. Provide clear, concise answers based on the ticket data provided."},
                    {"role": "user", "content": f"Based on this ticket data:\n{context}\n\nPlease answer this question: {query}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            # Fallback to basic analysis
            return analyze_tickets_basic(query, filtered_tickets, start_date, end_date)
            
    except Exception as e:
        logger.error(f"Error in AI analysis: {str(e)}")
        return f"Error analyzing tickets: {str(e)}"

# Update the sidebar query section with AI integration
st.sidebar.write("## Smart Query Assistant")
query_type = st.sidebar.radio("Select Query Type", ["Basic Analysis", "AI-Powered Analysis"])
user_query = st.sidebar.text_input(
    "Enter your query",
    help="""
    For Basic Analysis, try:
    - 'show status'
    - 'show category'
    - 'show count'
    
    For AI Analysis, ask anything in plain English, like:
    - 'What is the trend of tickets over time?'
    - 'Which category has the most urgent tickets?'
    - 'How many tickets were created last week?'
    """
)

if st.sidebar.button('Clear', key='clear_query'):
    st.session_state.query_input = ""
    
if st.sidebar.button('Analyze'):
    if user_query:
        st.write("### Analysis Result")
        
        with st.spinner('Analyzing tickets...'):
            if query_type == "Basic Analysis":
                answer = analyze_tickets_basic(user_query, open_tickets + closed_tickets, start_date, end_date)
            else:  # AI-Powered Analysis
                answer = analyze_tickets_with_ai(user_query, open_tickets + closed_tickets, start_date, end_date)
            
            # Display the result in a nice format
            st.markdown(f"""
            📊 **Query Result**
            ```
            {answer}
            ```
            """)
            
            # Show success animation
            st.balloons()
