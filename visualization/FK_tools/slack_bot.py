from slack_sdk import WebClient

import getpass

def send_slack_message(RECEIVER="Florian Kämpf",MESSAGE="Script finished!"):
    slack_token = "xoxb-2212881652034-3363495253589-2kSTt6BcH3YTJtb3hIjsOJDp"
    client = WebClient(token=slack_token)
    ul = client.users_list()
    ul['real_name']
    member_list = []



    for users in ul.data["members"]:
        member_list.append(users["profile"]['real_name'])
        if RECEIVER in users["profile"]['real_name']:
            chat_id = users["id"]

    client.conversations_open(users=chat_id)
    response = client.chat_postMessage(
        channel=chat_id,
        text=MESSAGE
    )