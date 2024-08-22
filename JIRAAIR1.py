import requests
from openai import organization
from requests.auth import HTTPBasicAuth
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta, timezone
import openai

# JIRA configuration
JIRA_URL = 'https://rfxdev.atlassian.net/rest/api/2/search'
USERNAME = 'akumar@rfxcel.com'

openai.api_key = OPENAI_API_KEY

def categorize_ticket(summary, description):
    categories = {
        "Aggregate": {"aggregate": 3, "aggregation": 3, "batch processing": 3, "lot": 3},
        "EPCIS": {"epcis": 4, "event capture": 4, "tracking": 4, "serialization": 4, "traceability": 4, "events": 4},
        "MDN": {"mdn": 2, "message disposition": 2, "acknowledgment": 2, "receipt": 2, "edi": 2, "message": 2},
        "Shipping": {"shipping": 5, "shipment": 5, "ship": 5, "carrier": 5, "freight": 5, "logistics": 5,
                     "transportation": 5, "delivery": 5}
    }

    ticket_text = summary.lower() + description.lower()
    category_scores = {category: sum(ticket_text.count(keyword) * weight for keyword, weight in keywords.items())
                       for category, keywords in categories.items()}

    # Find the category with the highest weighted score
    highest_category = max(category_scores, key=category_scores.get)
    highest_score = category_scores[highest_category]

    # Ensure only one category is set as "True", others are "False"
    if highest_score > 0:
        categorized = {category: "TRUE" if category == highest_category else "FALSE"
                       for category in categories}
    else:
        categorized = {category: "FALSE" for category in categories}

    return highest_category, categorized


def fetch_jira_tickets(jql):
    auth = HTTPBasicAuth(USERNAME, API_TOKEN)
    headers = {'Accept': 'application/json'}
    ticket_data = []
    start_at = 0
    batch_size = 100

    while True:
        params = {
            'jql': jql,
            'startAt': start_at,
            'maxResults': batch_size,
            'fields': 'key,summary,priority,status,resolution,created,updated,resolved,description,severity,priority,organization'
        }

        response = requests.get(JIRA_URL, headers=headers, params=params, auth=auth)

        if response.status_code == 200:
            tickets = response.json()['issues']
            for ticket in tickets:
                summary = ticket['fields'].get('summary', 'No summary provided')
                description = ticket['fields'].get('description', 'No description provided')
                description = description if description else 'No description provided'
                created = make_timezone_aware(str(ticket['fields']['created']))
                updated = make_timezone_aware(str(ticket['fields']['updated']))
                severity = ticket['fields'].get('severity', 'Unknown')
                #customer = ticket['fields'].get('customer', 'no customer provided')
                organization = ticket['fields'].get('organization', 'no organization provided')
                priority = ticket['fields'].get('priority', {}).get('name', 'no priority provided')
                # Categorize ticket using the keyword search
                category = categorize_ticket(summary, description)
                resolved = ticket['fields'].get('resolved', None)

                ticket_info = {
                    'key': ticket['key'],
                    'summary': summary,
                    'description': description[:100],
                    'priority': ticket['fields'].get('priority', {}).get('name', 'Unknown'),
                    'status': ticket['fields']['status']['name'],
                    'resolution': ticket['fields']['resolution']['name'] if ticket['fields']['resolution'] else 'Unresolved',
                    'created': created,
                    'updated': updated,
                    'category': category,
                    'severity': severity,
                    #'customer': customer,
                    'organization': organization,
                    'priority': priority,
                    'resolved': resolved
                }
                ticket_data.append(ticket_info)

            if len(tickets) < batch_size:
                break
            start_at += batch_size
        else:
            st.error(f"Failed to fetch data from JIRA: {response.status_code} {response.text}")
            break

    return ticket_data

def make_timezone_aware(date_input):
    if isinstance(date_input, datetime):
        return date_input.astimezone(timezone.utc)
    try:
        dt = datetime.strptime(date_input, '%Y-%m-%dT%H:%M:%S.%f%z')
    except ValueError:
        dt = datetime.strptime(date_input, '%Y-%m-%dT%H:%M:%S%z')
    return dt.astimezone(timezone.utc)

def display_high_level_info(open_tickets, closed_tickets, start_date, end_date):
    filtered_open_tickets = [ticket for ticket in open_tickets if
                             start_date <= make_timezone_aware(ticket['created']) <= end_date]
    filtered_closed_tickets = [ticket for ticket in closed_tickets if
                               start_date <= make_timezone_aware(ticket['updated']) <= end_date]

    new_tickets = len(filtered_open_tickets)
    closed_tickets_count = len(filtered_closed_tickets)
    pending_tickets = len(filtered_open_tickets)

    st.write("### Summary")
    high_level_info = pd.DataFrame({
        "New Tickets": [new_tickets],
        "Closed Tickets": [closed_tickets_count],
        #"Pending Tickets": [pending_tickets]
    })
    st.table(high_level_info.style.applymap(lambda val: 'color: red' if val > 0 else 'color: green'))

def display_tickets(ticket_data, start_date, end_date, selected_status):
    st.write("### Detailed Chart of Support Tickets")
    filtered_tickets = [
        ticket for ticket in ticket_data
        if start_date <= make_timezone_aware(ticket['created']) <= end_date
           and (ticket['status'] == selected_status or selected_status == "All")
    ]
    st.write(f"Showing {len(filtered_tickets)} tickets")
    if filtered_tickets:
        st.table([
            {
                "Actual Ticket Number": ticket['key'],
                "Title": ticket['summary'],
                "Created Date": ticket['created'],
                "Category": ticket['category']  # Include the category in the table
            } for idx, ticket in enumerate(filtered_tickets)
        ])

def display_charts(open_tickets, closed_tickets, start_date, end_date):
    filtered_open_tickets = [ticket for ticket in open_tickets if
                             start_date <= make_timezone_aware(ticket['created']) <= end_date]
    filtered_closed_tickets = [ticket for ticket in closed_tickets if
                               start_date <= make_timezone_aware(ticket['updated']) <= end_date]

    df_open = pd.DataFrame(filtered_open_tickets)
    df_closed = pd.DataFrame(filtered_closed_tickets)
    df_open['created_date'] = pd.to_datetime(df_open['created']).dt.date
    df_closed['updated_date'] = pd.to_datetime(df_closed['updated']).dt.date

    created_counts = df_open.groupby('created_date').size()
    closed_counts = df_closed.groupby('updated_date').size()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=created_counts.index, y=created_counts.values, name='Tickets Created', marker_color='red'))
    fig.add_trace(go.Scatter(x=closed_counts.index, y=closed_counts.values, name='Tickets Closed', marker_color='green',
                             yaxis='y2'))

    fig.update_layout(
        title='Tickets Created and Closed Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Tickets Created'),
        yaxis2=dict(title='Tickets Closed', overlaying='y', side='right')
    )

    st.plotly_chart(fig)

def visualize_categorized_tickets(open_tickets, closed_tickets, start_date, end_date):
    # Filter tickets based on the date range
    filtered_open_tickets = [ticket for ticket in open_tickets if
                             start_date <= make_timezone_aware(ticket['created']) <= end_date]
    filtered_closed_tickets = [ticket for ticket in closed_tickets if
                               start_date <= make_timezone_aware(ticket['updated']) <= end_date]

    # Combine filtered open and closed tickets
    filtered_tickets = filtered_open_tickets + filtered_closed_tickets

    df = pd.DataFrame(filtered_tickets)
    # Apply categorize_ticket function to each ticket
    df['category'] = df.apply(lambda row: categorize_ticket(row['summary'], row['description'])[0], axis=1)
    category_counts = df['category'].value_counts()
    category_colors = {
        "Aggregate": "blue",
        "EPCIS": "green",
        "MDN": "red",
        "Shipping": "purple",
        "Others": "gray"
    }

    colors = [category_colors.get(category, "gray") for category in category_counts.index]
    fig = go.Figure(data=[go.Bar(x=category_counts.index, y=category_counts.values, marker_color=colors, text=category_counts.values, textposition='auto')])
    fig.update_layout(
        title='Ticket Categories Distribution',
        xaxis_title='Category',
        yaxis_title='Number of Tickets',
        bargap=0.2,  # Adjusts the gap between bars
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'),  # Add grid lines for better readability
    )
    st.plotly_chart(fig)
def query_llm(question, context, open_ticket_statuses, closed_ticket_statuses):
    limited_context = "\n\n".join(context.split("\n\n")[:100000])

    open_statuses_str = ", ".join(open_ticket_statuses)
    closed_statuses_str = ", ".join(closed_ticket_statuses)

    prompt = (
        "You are an AI assistant specialized in analyzing JIRA tickets. Based on the following ticket descriptions and categories, "
        "please provide a consistent analysis and answer the user's question in a precise and unambiguous manner.\n\n"
        f"Context:\n{limited_context}\n\n"
        f"Open Ticket Statuses: {open_statuses_str}\n"
        f"Closed Ticket Statuses: {closed_statuses_str}\n\n"
        f"category:\n{categorize_ticket}\n\n"
        f"Question:\n{question}\n\n"
        "Please provide a consistent and accurate response:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides insights based on JIRA ticket descriptions & summary."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        temperature=0.1,
        n=1,
        stop=None,
        top_p=0.9
    )
    return response['choices'][0]['message']['content'].strip()

    # Check if the user query asks for visualization
    if "visualize" in question.lower() or "chart" in question.lower() or "show" in question.lower() or "display" in question.lower() or "plot" in question.lower() or "distribution" in question.lower() or "category" in question.lower() or "categorize" in question.lower():
        st.write("### Visualization of Categorized Tickets")
        visualize_categorized_tickets(ticket_data)
    return answer

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
    "Assigned for Advanced Resolution",
    "Assigned for Initial Evaluation",
    "Assigned for Resolution",
    "Escalated",
    "Evaluation In-Progress",
    "Hold for CSM Input",
    "On Hold - Customer",
    "On Hold - Scheduled Intervention",
    "Resolution In-Progress",
    "Resolution In-Progress (Escalated)",
    "Resolution InProgress",
    "Triage"
]
closed_ticket_statuses = [
    "Resolution Provided",
    "Resolved",
]
open_ticket_statuses_str = '","'.join(open_ticket_statuses)
open_tickets_jql = f'project = "CSR" AND status in ("{open_ticket_statuses_str}") AND created >= "{start_date.strftime("%Y-%m-%d")}" AND created <= "{end_date.strftime("%Y-%m-%d")}" ORDER BY created DESC'
closed_ticket_statuses_str = '","'.join(closed_ticket_statuses)
closed_tickets_jql = f'project = "CSR" AND status in ("{closed_ticket_statuses_str}") AND updated >= "{start_date.strftime("%Y-%m-%d")}" AND updated <= "{end_date.strftime("%Y-%m-%d")}" ORDER BY updated DESC'

open_tickets = fetch_jira_tickets(open_tickets_jql)
closed_tickets = fetch_jira_tickets(closed_tickets_jql)

status_options = ["All"] + sorted(set(ticket['status'] for ticket in open_tickets + closed_tickets))
selected_status = st.sidebar.selectbox("Select Status", status_options)
if st.sidebar.button('Fetch Data'):
    display_high_level_info(open_tickets, closed_tickets, start_date, end_date)
    #display_tickets(open_tickets, start_date, end_date, selected_status)
    display_charts(open_tickets, closed_tickets, start_date, end_date)
    visualize_categorized_tickets(open_tickets, closed_tickets, start_date, end_date)

st.sidebar.write("## Natural Language Query")
user_query = st.sidebar.text_input("Enter your query about JIRA tickets")
if st.sidebar.button('Clear', key='clear_query'):
    st.session_state.query_input = ""
if st.sidebar.button('Ask'):
    context = "\n\n".join([f"{ticket['key']}: {ticket['description']} "
                           f"Summary: {ticket['summary']} "
                           #f"Category: {ticket['category']} "
                           f"(Created: {ticket['created']}, Status: {ticket['status']}) "
                           f"(resolved: {ticket['resolved']}),status: {ticket['status']}"
                           f"Organization: {ticket['organization']}"
                           f"categorize_ticket: {categorize_ticket(ticket['summary'], ticket['description'])}"
                           for ticket in open_tickets + closed_tickets])
    if user_query:
        answer = query_llm(user_query, context, open_ticket_statuses, closed_ticket_statuses)
        st.write("### Answer to your query")
        st.write(answer)
        st.balloons()

# Final code = Working very well. Dont disturb it.

# to do = 1. Add the organization column in the table
         # 2. To add the custom column number of Severity column
