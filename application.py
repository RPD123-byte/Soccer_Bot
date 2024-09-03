import openai
import csv
import pandas as pd
import re
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from flask import request
from flask_cors import CORS
from transformers import GPT2Tokenizer
import textwrap
import builtins
import time
import ast
import os


openai_api_key = "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"

# Initialize Flask application
application = Flask(__name__)
CORS(application, resources={r"/ask": {"origins": "http://127.0.0.1:5500"}})
client = openai.OpenAI(api_key=openai_api_key) # be sure to set your OPENAI_API_KEY environment variable


Session(application)

@application.route('/')
def home():
    return render_template('chat.html')

# Custom print function
def wprint(*args, width=70, **kwargs):
    wrapper = textwrap.TextWrapper(width=width)
    wrapped_args = [wrapper.fill(str(arg)) for arg in args]
    builtins.print(*wrapped_args, **kwargs)

# Function to execute a thread and retrieve the completion
def get_completion(message, agent, funcs, thread, client):
    # Create new message in the thread
    message_response = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message
    )

    # Run the thread
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id,
    )

    while True:
        # Wait until run completes
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        if run.status in ['queued', 'in_progress']:
            time.sleep(1)
            continue

        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                print(f"Debug: Calling function {tool_call.function.name}", flush=True)

                wprint(f'\033[31mFunction: {tool_call.function.name}\033[0m')
                func = next((f for f in funcs if f.__name__ == tool_call.function.name), None)
                if func:
                    try:
                        # Assuming arguments are parsed correctly
                        func_instance = func(**eval(tool_call.function.arguments))  # Consider safer alternatives to eval
                        output = func_instance.run()

                        # Ensure output is a string
                        if not isinstance(output, str):
                            output = str(output)
                    except Exception as e:
                        output = f"Error: {e}"
                else:
                    output = "Function not found"
                wprint(f"\033[33m{tool_call.function.name}: {output}\033[0m")
                tool_outputs.append({"tool_call_id": tool_call.id, "output": output})

            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
        elif run.status == "failed":
            raise Exception(f"Run Failed. Error: {run.last_error}")
        else:
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            latest_message = messages.data[0].content[0].text.value
            return latest_message

class string_to_dict_tool:
    openai_schema = {
        "name": "string_to_dict_tool",
        "description": "Converts a string representation of a dictionary into a Python dictionary and writes it to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "criteria": {"type": "string", "description": "String representation of the dictionary"}
            },
            "required": ["criteria"]
        }
    }

    def __init__(self, criteria):
        self.criteria = criteria

    def run(self):
        try:
            # Convert the input string to a Python dictionary
            dict_obj = ast.literal_eval(self.criteria)

            # Write the dictionary to a text file
            with open('criteria.txt', 'w') as file:
                file.write(str(dict_obj))

            return {"output": "criteria.txt created successfully"}

        except (SyntaxError, ValueError) as e:
            return {"error": f"Error converting string to dictionary: {str(e)}"}
        
search_and_retrieve_tools = [string_to_dict_tool]

search_and_retrieve_agent = client.beta.assistants.create(
    name='Search and Retrieve Agent',
    instructions="""
    As a search and retrieve agent, your job is to take a user question, format it into a database query, perform the search, and return the results of that query.
    First take the user question and extract the appropriate data from it to get a database query. The format should be as follows:
    "{'Title1': ('equals/over/under', ['value']), 'Title2': ('equals/over/under', ['value'])}"
    
    For example, this is a valid database query string:
    "{'Position': ('equals', ['CM']),'GBE': ('equals', ['No']), 'IQ': ('over', [6])}"

    These are the possible titles:
    ['Date', 'Player', 'Club', 'League', 'TM Link', 'MV', 'xTV', 'Position',
    'Age', 'EU', 'Contract', 'Technique', 'IQ', 'Personality', 'Power',
    'Speed', 'Tier (future)', 'Energy', 'Rating', 'Current', 'Future',
    'up or down', 'GBE', 'Scout report']

    These are the options for the values:
    The options for Title Rating are Low, Medium, and Big. The options for GBE are No, Yes, and Panel.
    The options for Titles MV, xTV, Age, Technique, IQ, Personality, Power, Speed, Tier (future), Energy, Rating, Current, Future, up or down are all numbers
    that will be given in the question. 

    Once you get the proper format, print it out.
    Then use 'string_to_dict_tool' to convert it into a list of dictionaries.
    For the 'criteria' parameter, input the string you created with all the critieria given in the user question. 
    If you get errors, that means you haven't formatted the sting properly, so make sure the string is formatted properly.
    Only run the 'string_to_dict_tool' once. The only circumstance in which you should run it again is if you think the string you gave is formatted incorrectly.
    Now end the run.
    """,
    model="gpt-4-1106-preview",
    tools=[{"type": "function", "function": string_to_dict_tool.openai_schema},

           ]
)

def convert_to_float(d):
    for key, value in d.items():
        # Check if value is a tuple or a list
        if isinstance(value, (tuple, list)):
            new_values = []
            for item in value:
                # If item is a list, iterate through its elements
                if isinstance(item, list):
                    new_list = []
                    for elem in item:
                        if isinstance(elem, str) and elem.isdigit():
                            new_list.append(float(elem))
                        else:
                            new_list.append(elem)
                    new_values.append(new_list)
                else:
                    # Keep non-list items as is
                    new_values.append(item)
            d[key] = tuple(new_values) if isinstance(value, tuple) else new_values
    return d

def read_and_convert(file_path='criteria.txt'):
    try:
        with open(file_path, 'r') as file:
            data_str = file.read()
            data_dict = ast.literal_eval(data_str)
        
        converted_data = convert_to_float(data_dict)

        # Write the updated dictionary back to the file
        with open(file_path, 'w') as file:
            file.write(str(converted_data))

        return converted_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

def search_players(criteria, df):
    filtered_df = df.copy()  # Make a copy of the DataFrame

    for column, (operator, values) in criteria.items():
        if column not in filtered_df.columns:
            continue  # Skip if the column doesn't exist in the DataFrame

        # Ensure values is a list
        if not isinstance(values, list):
            values = [values]

        # Filter out rows with NaN in the target column
        filtered_df = filtered_df[pd.notna(filtered_df[column])]

        # Try to convert values to the same data type as the column
        try:
            col_type = filtered_df[column].dtype
            if col_type == object:  # If column type is object, attempt string conversion
                values = [str(val) for val in values]
            else:
                values = [col_type.type(val) for val in values]
        except TypeError:
            continue  # Skip if type conversion fails

        # Apply filters based on the operator
        if operator == 'under':
            filtered_df = filtered_df[filtered_df[column] < max(values)]
        elif operator == 'over':
            filtered_df = filtered_df[filtered_df[column] > min(values)]
        elif operator == 'equals':
            filtered_df = filtered_df[filtered_df[column].isin(values)]

    return filtered_df

def read_criteria_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the content of the file
            content = file.read()

            # Convert the string back to a dictionary
            criteria = ast.literal_eval(content)
            return criteria

    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except SyntaxError as e:
        print(f"Syntax error in the file content: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def sort_players(players):
    columns_to_average = ['Technique', 'IQ', 'Personality', 'Power', 'Speed', 'Energy']

    # Convert columns to numeric, setting non-numeric values to NaN
    for col in columns_to_average:
        players[col] = pd.to_numeric(players[col], errors='coerce')

    # Calculate the mean, ignoring NaN values
    players['strength'] = players[columns_to_average].mean(axis=1, skipna=True)

    # Reorder the DataFrame based on the 'strength' column, from highest to lowest
    players = players.sort_values(by='strength', ascending=False)

    return players

def string_to_int(results_string):
    # Find all numbers in the string
    numbers = re.findall(r'\d+', results_string)

    # Assuming we want the first number found, convert it to an integer
    results_number = int(numbers[0]) if numbers else None

    return results_number

def get_number_of_results(question):
    system_instructions = """
                    You are going to take a user's question and check if they specified the number of results they want. If they have specified, return the
                    number of results like this:
                    '[number_of_results]'
                """
    user_prompt = f"""
                    Context:
                    {question}
                    A:
                """
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt}
            ]
        )
        # Extracting the response
        response = completion.choices[0].message.content.strip()

        # Try to convert the response to an integer
        number_of_results = string_to_int(response)
        return number_of_results

    except ValueError:
        # Handle the case where conversion to integer fails
        print("Could not extract a valid number of results.")
        return None
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
        return None
    
def get_score(question):
    system_instructions = (
        f"""
        Receive and understand the use question and determine a score from 1 to 5 for the question based on how higly players should be ranked or vice versa. 
        For example if asked for the best results the score would be 5 and if asked for the worst results the score would be 1.
        if question contain "best" or "top":
            return 5
        elif question contains "worst" or "bottom"
            return 1
        else:
            return somewhere between 2 and 4 based on what the user wants
        The Return format should just be a score with no reasoning like this:
        [insert score here]
        """
    )

    # User prompt including the Tar Heel Tracker text and the question
    user_prompt = f"""
                Context:
                {question}
                A:
                """

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ]
    )
    # Extracting the response
    response = completion.choices[0].message.content.strip()

    number_of_results = string_to_int(response)
    return number_of_results

def get_results_based_on_sentiment(players_df, sentiment_score, results_number):
    if len(players_df) > 10:
        # Ensure the sentiment score is within the expected range
        sentiment_score = max(1, min(sentiment_score, 5))

        # Define the number of results to return if not specified
        default_results_number = 10

        # Use the specified number of results or the default
        if results_number is not None:
            if results_number > 60:
                results_number = 20
        num_results = results_number if results_number is not None else default_results_number
        num_results = min(num_results, len(players_df))  # Ensure we don't exceed the DataFrame's length

        # Split the DataFrame into five equal segments
        segment_length = len(players_df) // 5
        segments = [players_df.iloc[i * segment_length: (i + 1) * segment_length] for i in range(5)]

        # Adjust the last segment to include any remaining rows
        if len(players_df) % 5 != 0:
            segments[-1] = players_df.iloc[4 * segment_length:]

        # Select the segment based on the sentiment score
        if sentiment_score == 1:
            # Return the bottom results
            return players_df.tail(num_results)
        else:
            # Best results (segment 4) for score 5, and so on
            selected_segment = segments[5 - sentiment_score]

        # Return the specified number of results from the selected segment
        return selected_segment.head(num_results)
    else:
        return players_df
    
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def calculate_token_count(conversations):
    return sum([len(tokenizer.encode(entry['content'])) for entry in conversations])

def final_answer(question, selected_players, conversation_history):
    # System instructions for the AI model
    system_instructions = (
        """ Understand the question given by the user and use the context given under Context: to answer the question. 
            Make sure to include the exact number of results requested by the user.
            All of the information in the given context should be present in the answer to the question.
            If you aren't given any information under to pull from that means that there are no players with the criteria specified by the user and you should return a message saying so.
        """
    )

    # Append new system and user messages to the conversation history
    conversation_history.append({"role": "system", "content": system_instructions})

    # Ensure the conversation history does not exceed token limits
    while calculate_token_count(conversation_history) > 16000:
        conversation_history.pop(0)

    # User prompt including the context and the question
    user_prompt = f"Context:\n{selected_players}\n__________________\nQ: {question}\nA:"

    conversation_history.append({"role": "user", "content": user_prompt})

    # Generate the response
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=conversation_history
    )

    # Update the conversation history with the latest response
    if 'choices' in response and len(response['choices']) > 0:
        choice = response['choices'][0]
        if 'message' in choice:
            conversation_history.append(choice['message'])

    # Extract only the AI's response to the latest question
    if response.choices:
        latest_response = response.choices[0].message.content.split('\nA:')[-1].strip()
    else:
        latest_response = "No response."

    # Return the response content and the updated conversation history
    return latest_response, conversation_history


@application.route('/ask', methods=['POST'])
def ask():
    df = pd.read_csv('modified_player_database.csv', low_memory=False)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_seq_items', None)
    pd.set_option('display.width', None)

    data = request.get_json()
    question = data['question']  
    # Create a new thread
    thread = client.beta.threads.create()
    user_input = question
    message = get_completion(user_input, search_and_retrieve_agent, search_and_retrieve_tools, thread, client)
    wprint(f"\033[34m{search_and_retrieve_agent.name}: {message}\033[0m")
    converted_data = read_and_convert()
    print("Updated data:", converted_data)
    file_path = 'criteria.txt'
    criteria = read_criteria_from_file(file_path)
    print(criteria)
    players = search_players(criteria, df)
    if players.empty:
        thread = client.beta.threads.create()
        user_input = question
        message = get_completion(user_input, search_and_retrieve_agent, search_and_retrieve_tools, thread, client)
        wprint(f"\033[34m{search_and_retrieve_agent.name}: {message}\033[0m")
        converted_data = read_and_convert()
        print("Updated data:", converted_data)
        file_path = 'criteria.txt'
        criteria = read_criteria_from_file(file_path)
        print(criteria)
        players = search_players(criteria, df)
    sorted_players = sort_players(players)
    print("working")
    results_number = get_number_of_results(question)
    sentiment_score = get_score(question)
    print(sentiment_score)
    final_results = get_results_based_on_sentiment(sorted_players, sentiment_score, results_number)
    print(final_results)
    conversation_history = []
    response, conversation_history = final_answer(question, final_results, conversation_history)
    print(response)
    return jsonify({"response": response}), 200


# Run the Flask application
if __name__ == '__main__':
    application.run(debug=True, port=4000)


