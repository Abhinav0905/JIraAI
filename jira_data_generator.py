import os
import random
from datetime import datetime, timedelta
from typing import List, Dict
import requests
from requests.auth import HTTPBasicAuth
import faker
import time

# Initialize Faker for generating realistic data
fake = faker.Faker()

# Jira Configuration
JIRA_URL = os.getenv('JIRA_URL', 'https://testpocjira.atlassian.net')
JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
JIRA_EMAIL = os.getenv('JIRA_EMAIL')
PROJECT_KEY = os.getenv('JIRA_PROJECT_KEY', 'KAN')

# Categories and their related keywords for more realistic ticket generation
CATEGORIES = {
    "Aggregate": [
        "data aggregation", "merge records", "combine datasets", "consolidate information",
        "aggregate report", "data compilation", "summary generation"
    ],
    "EPCIS": [
        "event data", "supply chain", "tracking", "traceability", "product movement",
        "EPCIS event", "visibility data", "chain of custody"
    ],
    "MDN": [
        "message delivery", "notification", "delivery status", "MDN processing",
        "receipt confirmation", "message tracking", "delivery report"
    ],
    "Shipping": [
        "shipment", "delivery", "logistics", "transportation", "carrier integration",
        "shipping label", "tracking number", "freight"
    ]
}

SEVERITIES = ["High", "Medium", "Low", "Critical"]
STATUSES = ["Open", "In Progress", "Under Review", "Testing", "Closed"]

def create_dummy_ticket() -> Dict:
    """Create a single dummy ticket with realistic data."""
    # Pick a random category and its associated keywords
    category = random.choice(list(CATEGORIES.keys()))
    keywords = CATEGORIES[category]
    
    # Generate summary using category context
    summary = f"{category}: {fake.sentence(ext_word_list=keywords)}"
    
    # Generate a more detailed description
    description = f"""
{fake.paragraph(nb_sentences=2, ext_word_list=keywords)}

Technical Details:
- Component: {category}
- Environment: {random.choice(['Production', 'Staging', 'Development'])}
- Browser: {random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'])}
- OS: {random.choice(['Windows 10', 'macOS', 'Linux'])}

Steps to Reproduce:
1. {fake.sentence()}
2. {fake.sentence()}
3. {fake.sentence()}

Expected Behavior:
{fake.sentence()}

Actual Behavior:
{fake.sentence()}
"""
    
    return {
        "fields": {
            "project": {"key": PROJECT_KEY},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"},
            "labels": [category]
        }
    }

def transition_issue(issue_key: str, target_status: str, auth: HTTPBasicAuth, headers: dict) -> bool:
    """Transition a Jira issue to the target status."""
    # First, get available transitions
    transitions_url = f"{JIRA_URL}/rest/api/2/issue/{issue_key}/transitions"
    response = requests.get(transitions_url, headers=headers, auth=auth)
    
    if response.status_code != 200:
        print(f"Failed to get transitions for {issue_key}: {response.text}")
        return False
    
    transitions = response.json()['transitions']
    target_transition = next((t for t in transitions if t['to']['name'].lower() == target_status.lower()), None)
    
    if not target_transition:
        print(f"No transition found to status '{target_status}' for {issue_key}")
        return False
    
    # Perform the transition
    transition_data = {
        "transition": {
            "id": target_transition['id']
        }
    }
    
    response = requests.post(
        transitions_url,
        json=transition_data,
        headers=headers,
        auth=auth
    )
    
    if response.status_code == 204:
        print(f"Successfully transitioned {issue_key} to {target_status}")
        return True
    else:
        print(f"Failed to transition {issue_key} to {target_status}: {response.text}")
        return False

def create_tickets(num_tickets: int = 500) -> None:
    """Create and push multiple tickets to Jira."""
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # Define status distribution
    statuses = {
        "To Do": 0.2,       # 20% of tickets
        "In Progress": 0.3,  # 30% of tickets
        "QA": 0.3,          # 30% of tickets
        "Done": 0.2         # 20% of tickets
    }
    
    success_count = 0
    created_issues = []
    
    for i in range(num_tickets):
        try:
            ticket_data = create_dummy_ticket()
            response = requests.post(
                f"{JIRA_URL}/rest/api/2/issue",
                json=ticket_data,
                headers=headers,
                auth=auth
            )
            
            if response.status_code == 201:
                success_count += 1
                issue_key = response.json()['key']
                created_issues.append(issue_key)
                print(f"Created ticket {i+1}/{num_tickets}")
            else:
                print(f"Failed to create ticket {i+1}: {response.text}")
            
            # Sleep to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error creating ticket {i+1}: {str(e)}")
    
    print(f"\nSuccessfully created {success_count} out of {num_tickets} tickets")
    
    # Transition tickets according to the distribution
    print("\nTransitioning tickets to their target statuses...")
    for issue_key in created_issues:
        # Determine target status based on distribution
        rand = random.random()
        cumulative = 0
        target_status = None
        
        for status, probability in statuses.items():
            cumulative += probability
            if rand <= cumulative:
                target_status = status
                break
        
        if target_status != "To Do":  # Default status doesn't need transition
            # For QA status, first move to In Progress
            if target_status == "QA":
                transition_issue(issue_key, "In Progress", auth, headers)
                time.sleep(1)  # Respect rate limits
            
            transition_issue(issue_key, target_status, auth, headers)
            time.sleep(1)  # Respect rate limits

if __name__ == "__main__":
    if not all([JIRA_API_TOKEN, JIRA_EMAIL]):
        print("Please set JIRA_API_TOKEN and JIRA_EMAIL environment variables")
    else:
        create_tickets(200)  # Creating 200 tickets as a test