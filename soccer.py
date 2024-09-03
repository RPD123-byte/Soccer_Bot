import openai
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from flask import request
from flask_cors import CORS
from transformers import GPT2Tokenizer





openai.api_key = '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'

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "http://127.0.0.1:5500"}})

Session(app)

def format_question(question):
    # System instructions for the AI model
    system_instructions = (
    """
    Take the user question and determine which titles are relevant to the question being asked and what value corresponds to that title
    The possible titles are:
    ['Date', 'Player', 'Club', 'League', 'TM Link', 'MV', 'xTV', 'Position',
    'Age', 'EU', 'Contract', 'Technique', 'IQ', 'Personality', 'Power',
    'Speed', 'Tier (future)', 'Energy', 'Rating', 'Current', 'Future',
    'up or down', 'GBE', 'Scout report']
    
    Make sure to understand the user's question as the titles or other might not be given in the exact format, so convert it as needed

    The values that correspond to the titles are given in the question. For the corresponding value for the position title the value should always
    be abbreviated. 
    The options for Rating are Low, Medium, and Big. The options for GBE are No, Yes, and Panel.
    The options for MV, xTV, Age, Technique, IQ, Personality, Power, Speed, Tier (future), Energy, Rating, Current, Future, up or down are all numbers
    that will be given in the question. Copy over the exact number given into the final result for these titles. These can only be numbers under no
    circumstances should these have any text or be in numeric format
    The options for Player are the names of the player. The options for Club and League are also the names of the club and league. These are always words,
    under no circumstances should they be numbers.
    Return it as:
    [title:value,title2:value, etc]

    It is possible to have any number of criteria so your response should change according to the number

    If only player name is given return it as 
    [Player:name]

    """
    )

    # User prompt including the Tar Heel Tracker text and the question
    user_prompt = f"""
                Context:
                {question}
                A:
                """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        max_tokens=2000 # Adjust based on how long you expect the answer to be
    )

    return response['choices'][0]['message']['content'].strip()

def convert_string_to_dict(input_string, string_keys, integer_keys):
    # Remove brackets and split the string by commas
    items = input_string.strip('[]').split(', ')

    # Initialize an empty dictionary
    result_dict = {}

    for item in items:
        # Split each item by colon
        key_value = item.split(':')
        if len(key_value) == 2:
            key, value = key_value
            # Remove any extra quotes
            key = key.strip()
            value = value.strip()
            
            # Extract only digits from the string and convert to integer if possible
            if re.search(r'\d', value):  # Check if there are any digits in the string
                numeric_value = int(re.sub(r'\D', '', value))  # Remove non-digit characters and convert to int
                result_dict[key] = numeric_value
            else:
                result_dict[key] = value

    # Check for wrong entries and remove them
    for key in list(result_dict.keys()):
        if (key in string_keys and not isinstance(result_dict[key], str)) or \
           (key in integer_keys and not isinstance(result_dict[key], int)):
            del result_dict[key]

    return result_dict

def find_similar_values(df, column, query, top_n=10):
    unique_values = df[column].dropna().unique()
    query_lower = query.lower()

    # Compare the query with unique values (considering partial matches) without altering their case
    similar_values = [val for val in unique_values if query_lower in str(val).lower()][:top_n]
    return similar_values

# Process each item in the dictionary
def process_items(result_dict, df):
    output_dict = {}
    for title, query in result_dict.items():
        if title in df.columns:
            if isinstance(query, str):
                # Process string values
                similar_values = find_similar_values(df, title, query)
                output_dict[f"{title}"] = similar_values
            elif isinstance(query, int):
                # Directly return integer values
                output_dict[f"{title}"] = [query]
    return output_dict


# Function to remove unwanted variants from the dictionary values
def remove_unwanted_variants(input_dict):
    unwanted_variants = ['Not Found', 'No Data']
    cleaned_dict = {}
    for key, values in input_dict.items():
        if isinstance(values, list):
            # Filter out unwanted variants from lists
            cleaned_values = [value for value in values if value not in unwanted_variants]
            cleaned_dict[key] = cleaned_values
        else:
            # Keep single values as is
            cleaned_dict[key] = values
    return cleaned_dict

def final_formatting_for_criteria(question, output_dict):
    # System instructions for the AI model
    system_instructions = (
        f"""
        Convert the question from the user into this format:
        criteria = [
            'Title': ('equals/over/under', ["string" or integer, "string2" or integer2, etc]),
            'Title': ('equals/over/under', ["string" or integer, "string2" or integer2, etc]),
            'Title': ('equals/over/under', ["string" or integer, "string2" or integer2, etc])
        ]

        You can have only one searcher per title or you can have more.

       Make sure to understand the user's question as the titles or other might not be given in the exact format, so convert it as needed       
       These are the options for titles you should consider. The keys in the dictionary contain the name of the title and the values are the options
       {output_dict}
       Make sure to include all the options that are relevant and to the user query in the result. Options should be considered relevant if they contain
       keywords from the query or seem semantically similar. Don't blindly include everything think about which make sense to include and which don't. If any
       of the keywords says No Data or Not Found or -, then don't include them.
       """
    )

    # User prompt including the Tar Heel Tracker text and the question
    user_prompt = f"""
                Context:
                {question}
                A:
                """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        max_tokens=2000 # Adjust based on how long you expect the answer to be
    )

    return response['choices'][0]['message']['content'].strip()

def parse_criteria_string(input_str):
    # Removing "criteria = [" and "]" from the string
    formatted_str = input_str.replace("criteria = [", "").replace("]", "").strip()

    # Splitting into individual criteria
    criteria_parts = formatted_str.split("),\n    ")

    criteria_dict = {}

    for part in criteria_parts:
        # Split part into key and value
        key, value = part.split(": (", 1)
        key = key.strip("'")

        # Clean and split the value part
        condition, values_str = value.rstrip(')').split(", [", 1)
        condition = condition.strip("'")
        
        # Convert the string of values into a list
        values = [v.strip().strip("'") for v in values_str.strip("[]").split(", ")]
        
        # Convert numeric strings to integers or floats, and remove brackets if they exist
        processed_values = []
        for v in values:
            if v.isdigit():
                processed_values.append(int(v))
            elif v.replace('.', '', 1).isdigit():
                processed_values.append(float(v))
            else:
                processed_values.append(v)

        # Remove outer brackets for single-element lists
        if len(processed_values) == 1:
            criteria_dict[key] = (condition, processed_values[0])
        else:
            criteria_dict[key] = (condition, processed_values)

    return criteria_dict

def number_of_results(question):
    # System instructions for the AI model
    system_instructions = (
        f"""
        You are going to take a user's question and check if they specified the number of results they want. If they have specified return the
        number of results like this:
        [number_of_results]
        """
    )

    # User prompt including the Tar Heel Tracker text and the question
    user_prompt = f"""
                Context:
                {question}
                A:
                """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        max_tokens=2000 # Adjust based on how long you expect the answer to be
    )

    return response['choices'][0]['message']['content'].strip()


def string_to_int(results_string):
    # Find all numbers in the string
    numbers = re.findall(r'\d+', results_string)

    # Assuming we want the first number found and convert it to an integer
    results_number = int(numbers[0]) if numbers else None

    return results_number

def search_players(criteria, df):
    filtered_df = df.copy()  # Make a copy of the DataFrame
    for column, (operator, values) in criteria.items():
        if column not in filtered_df.columns:
            continue  # Skip if the column doesn't exist in the DataFrame

        # Ensure values is a list
        if not isinstance(values, list):
            values = [values]

        # Convert values to the appropriate data type of the column
        col_type = filtered_df[column].dtype
        values = [col_type.type(val) for val in values]

        if operator == 'under':
            filtered_df = filtered_df[filtered_df[column] <= max(values)]
        elif operator == 'over':
            filtered_df = filtered_df[filtered_df[column] >= min(values)]
        elif operator == 'equals':
            filtered_df = filtered_df[filtered_df[column].isin(values)]

    return filtered_df

# def modify_players(players, results_number):
#     columns_to_average = ['Technique', 'IQ', 'Personality', 'Power', 'Speed', 'Energy']
#     players['strength'] = players[columns_to_average].mean(axis=1)

#     # Reorder the DataFrame based on the 'strength' column, from highest to lowest
#     players = players.sort_values(by='strength', ascending=False)
#     players = players.drop("strength", axis = 1)
#     if results_number is not None:
#         players = players[0:results_number]

#     return players

def sort_players(players):
    columns_to_average = ['Technique', 'IQ', 'Personality', 'Power', 'Speed', 'Energy']

    # Convert columns to numeric, setting non-numeric values to NaN
    for col in columns_to_average:
        players[col] = pd.to_numeric(players[col], errors='coerce')

    # Calculate the mean, ignoring NaN values
    players['strength'] = players[columns_to_average].mean(axis=1, skipna=True)

    # Reorder the DataFrame based on the 'strength' column, from highest to lowest
    players = players.sort_values(by='strength', ascending=False)
    
    # Optionally drop the 'strength' column if not needed
    # players = players.drop("strength", axis=1)

    # Limit the number of results if specified
    # if results_number is not None and results_number > 5 and results_number <= 50:
    #     players = players.head(results_number)
    # elif results_number > 50:
    #     players = players.head(10)
    # else:
    #     players = players.head(10)

    return players

def get_score(question):
    # System instructions for the AI model
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

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        max_tokens=2000 # Adjust based on how long you expect the answer to be
    )

    return response['choices'][0]['message']['content'].strip()

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

# def final_answer(question, selected_players):
#     # System instructions for the AI model
#     system_instructions = (
#     f"""
#     Understand the question given by the user and use the context given under Context: to answer the question
#     """
#     )

#     # User prompt including the Tar Heel Tracker text and the question
#     user_prompt = f"""
#                 Context:
#                 {selected_players}
#                 __________________
#                 Q: {question}
#                 A:
#                 """

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-1106",
#         messages=[
#             {"role": "system", "content": system_instructions},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.5,
#         max_tokens=2000 # Adjust based on how long you expect the answer to be
#     )

#     return response['choices'][0]['message']['content'].strip()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def calculate_token_count(conversations):
    return sum([len(tokenizer.encode(entry['content'])) for entry in conversations])

def final_answer(question, selected_players, conversation_history):
    # System instructions for the AI model
    system_instructions = (
    f"""
    Understand the question given by the user and use the context given under Context: to answer the question
    """
    )

    # Append new system and user messages to the conversation history
    conversation_history.append({"role": "system", "content": system_instructions})

    # Ensure the conversation history does not exceed token limits
    while calculate_token_count(conversation_history) > 16000:
        conversation_history.pop(0)

    # User prompt including the Tar Heel Tracker text and the question
    user_prompt = f"""
                Context:
                {selected_players}
                __________________
                Q: {question}
                A:
                """
    
    conversation_history.append({"role": "user", "content": user_prompt})

    # Include the conversation history in the response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=conversation_history,
        temperature=0.1,
        max_tokens=2000 # Adjust based on how long you expect the answer to be
    )

    # Update the conversation history with the latest response
    if 'choices' in response and len(response['choices']) > 0:
        choice = response['choices'][0]
        if 'message' in choice:
            conversation_history.append(choice['message'])

    # Return the response content and the updated conversation history
    return response['choices'][0]['message']['content'].strip(), conversation_history




@app.route('/ask', methods=['POST'])
def ask():
    df = pd.read_csv('modified_player_database.csv', low_memory=False)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_seq_items', None)
    pd.set_option('display.width', None)

    data = request.get_json()
    question = data['question']    
    input_question = format_question(question)
    string_keys = ['Player', 'Club', 'League', 'Date', 'TM Link', 'Position', 'EU', 'Contract', 'Rating', 'GBE', 'Scout Report']
    integer_keys = ['xTV','MV', 'Age', 'Technique', 'IQ', 'Personality', 'Power', 'Speed', 'Tier (future)', 'Energy', 'Current', 'Future', 'up or down']
    result_dict = convert_string_to_dict(input_question, string_keys, integer_keys)
    print(result_dict)
    output_dict = process_items(result_dict, df)
    print(output_dict)
    output_dict = remove_unwanted_variants(output_dict)
    criteria_string = final_formatting_for_criteria(question, output_dict)
    criteria = parse_criteria_string(criteria_string) 
    print(criteria) 
    results_string = number_of_results(question)
    print(results_string)
    results_number = string_to_int(results_string)
    print(results_number)
    players = search_players(criteria, df)
    sentiment_score = get_score(question)
    sentiment_score = string_to_int(sentiment_score)
    print(sentiment_score)
    sorted_players = sort_players(players)
    # print(sorted_players)
    sorted_players = get_results_based_on_sentiment(sorted_players, sentiment_score, results_number)
    if 'conversation_history' not in globals():
        # If not defined, initialize it as an empty list
        conversation_history = []

    response, conversation_history = final_answer(question, sorted_players, conversation_history)
    print(response)
    # players_dict = sorted_players.where(pd.notnull(sorted_players), None).to_dict(orient='records')
    return jsonify({"response": response}), 200



# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=8000)