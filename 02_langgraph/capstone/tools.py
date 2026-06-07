"""
capstone/tools.py — the data lookups the specialist may call.

These are canned (fake) on purpose: in a workshop you don't want real systems in
the loop. Each returns a short string the model can read back to the customer.
Swap the bodies for real API calls later — the graph never has to change.

Remember: the DOCSTRING is what the model reads to decide when to call a tool.
"""

from langchain_core.tools import tool


@tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order by its order id (e.g. 'A1234')."""
    fake = {
        "a1234": "Order A1234: shipped, arriving tomorrow by 5pm.",
        "b5678": "Order B5678: payment failed, not yet shipped.",
    }
    return fake.get(order_id.lower(), f"No order found with id {order_id}.")


@tool
def get_billing_summary(account_id: str) -> str:
    """Get recent charges for an account id (e.g. 'ACC-9'). Use for billing questions."""
    fake = {
        "acc-9": "ACC-9 last charges: $20 on Jun 1, $20 on Jun 1 (DUPLICATE), $20 on May 1.",
        "acc-1": "ACC-1 last charges: $9.99 on Jun 3. No duplicates.",
    }
    return fake.get(account_id.lower(), f"No billing record for account {account_id}.")


@tool
def search_help_center(query: str) -> str:
    """Search help-center articles for how-to / troubleshooting questions."""
    articles = {
        "reset password": "To reset your password: Settings > Security > Reset, "
                          "then follow the email link (valid 30 min).",
        "export data": "Export your data from Settings > Privacy > Export; "
                       "you'll get a download link within an hour.",
        "cancel": "To cancel a subscription: Billing > Plan > Cancel. "
                  "Access continues until the period ends.",
    }
    q = query.lower()
    for key, text in articles.items():
        if key in q:
            return text
    return "No exact article found. Suggest contacting support for specifics."


# The specialist node binds this whole list. Add/remove tools freely.
SUPPORT_TOOLS = [get_order_status, get_billing_summary, search_help_center]
TOOLS_BY_NAME = {t.name: t for t in SUPPORT_TOOLS}
