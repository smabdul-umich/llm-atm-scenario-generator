import streamlit as st
from streamlit_feedback import streamlit_feedback
import pandas as pd
from prompts import *
from langchain_openai import AzureChatOpenAI
import os
import re
from dotenv import load_dotenv
import datetime
from io import StringIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Set the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath("app.py")))
feedback_output_file = "feedback.csv"
NUM_RESPONSES = 3
MAX_NUM_MESSAGES = 8

# Load environment file for secrets.
try:
    if load_dotenv('env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()

# Define llm parameters
llm = AzureChatOpenAI(
    deployment_name=os.environ['model'],  # e.g. gpt-35-turbo
    openai_api_version=os.environ['API_VERSION'],  # e.g. 2023-05-15
    openai_api_key=os.environ['OPENAI_API_KEY'],  # secret
    azure_endpoint=os.environ['openai_api_base'],  # a URL
    openai_organization=os.environ['OPENAI_organization']  
)

wx_context = """
Here is weather data you should use in the training script you generate for EWR. In the training script you generate, maintain the same format as in the example scripts, and make sure the script you generate is complete:
Locid	Date	Local_hour	Wind_Dir	Temp (F)	Wind_Speed	Ceiling	Visibility	Airport_WX	Nearby_TS	Enroute_TS
EWR	5/30/18	0	100	73	4	999	10		0	0
EWR	5/30/18	1	100	71	8	999	10		0	0
EWR	5/30/18	2	100	68	6	999	10		0	0
EWR	5/30/18	3	60	62	4	4	1.5		0	0
EWR	5/30/18	4	60	64	4	3	2	BR	0	0
EWR	5/30/18	5	70	64	4	2	2	BR	0	0
EWR	5/30/18	6	70	64	4	2	2	BR	0	0
EWR	5/30/18	7	80	64	4	3	2	#NAME?	0	0
EWR	5/30/18	8	90	66	4	5	2	#NAME?	0	0
EWR	5/30/18	9	110	66	5	6	2		0	0
EWR	5/30/18	10	VRB	69	3	8	10		0	0
EWR	5/30/18	11	80	69	8	11	10		0	0
EWR	5/30/18	12	80	69	7	15	10		0	0
EWR	5/30/18	13	140	69	7	16	10		0	0
EWR	5/30/18	14	180	68	8	13	10		0	0
EWR	5/30/18	15	180	68	8	13	10		0	0
EWR	5/30/18	16	190	66	7	250	10		0	0
EWR	5/30/18	17	190	66	7	250	10		0	0
EWR	5/30/18	18	140	59	8	6	10		0	0
EWR	5/30/18	19	120	59	7	3	7		0	0
EWR	5/30/18	20	120	59	7	3	7		0	0
EWR	5/30/18	21	0	60	0	4	6	#NAME?	0	0
EWR	5/30/18	22	0	62	0	7	10		0	0
EWR	5/30/18	23	70	62	4	32	9		0	0
"""

def query_gpt(training_prompt):
    """
    Generate training scripts based on the provided prompt.
    """
    prompt = instruction_prompt + "\n\n".join(
        query_template.format(nl_query=ex[0], sql_query=ex[1]) for ex in few_shot_examples
    )
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", prompt + "\n\n" + semantic_parsing_instruction + f" {training_prompt}" + wx_context),
    ]

    # Generate NUM_RESPONSES training scripts
    training_scenarios = [None for _ in range(NUM_RESPONSES)]
    for idx in range(NUM_RESPONSES):
        response = llm.invoke(messages)
        response_text = response.content
        parsed_response = re.search(r'```([\s\S]*?)```', response_text)
        parsed_response = parsed_response.group(1).strip()
        training_scenarios[idx] = parsed_response

    return training_scenarios

# Initialize session state variables
if 'data' not in st.session_state:
    if os.path.exists(feedback_output_file):
        st.session_state.data = pd.read_csv(feedback_output_file)
    else:
        st.session_state.data = pd.DataFrame(columns=['Timestamp', 'Query', 'Output', 'Type', 'Score', 'Feedback Text'])

if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0

if 'training_scenarios' not in st.session_state:
    st.session_state.training_scenarios = [None for _ in range(NUM_RESPONSES)]

# Define your function that processes the query
def process_query(query):
    return query_gpt(query)

# Normalize the text as necessary
def normalize_feedback_text(feedback_text):
    if feedback_text is None: return ""
    return feedback_text.strip()

def add_feedback(feedback, option_idx=0):
    normalized_feedback = normalize_feedback_text(feedback["text"])
    new_row = (
        str(datetime.datetime.now()),  # Timestamp
        query,
        st.session_state.training_scenarios[option_idx],
        feedback["type"],
        feedback["score"],
        normalized_feedback,
    )

    # Append feedback to the session state dataframe
    st.session_state.data = pd.concat(
        [st.session_state.data, pd.DataFrame([new_row], columns=st.session_state.data.columns)],
        ignore_index=True
    )

    # Update session state to reflect feedback submission
    st.session_state.feedback_given[option_idx] = True
    st.session_state.feedback_key += 1
    st.session_state.feedback_text_given[option_idx] = False

# Streamlit app layout
st.title('Training Scenario Generator (Beta)')

# Use a form to trigger actions on "Enter"
with st.form(key='query_form'):
    query = st.text_input('Enter your training scenario prompt:')
    submit_button = st.form_submit_button('Submit')

if submit_button:
    with st.spinner('Generating training scenarios...'):
        # Process the query and get the output
        training_scenarios = process_query(query)
        st.session_state.training_scenarios = training_scenarios  # Store output in session state
        st.session_state.last_query = query
        st.session_state.feedback_given = [False for _ in training_scenarios]
        st.session_state.feedback_text_given = [False for _ in training_scenarios]

# Add spacing after the input form and before displaying the output
st.markdown("<br>", unsafe_allow_html=True)

# Display the output if it exists
if st.session_state.training_scenarios[0] is not None:
    st.markdown("### Generated Training Scenarios")
    tabs = st.tabs([f"Option {idx}" for idx in range(NUM_RESPONSES) if idx < len(st.session_state.training_scenarios) and st.session_state.training_scenarios[idx]])
    for idx, training_scenario in enumerate(st.session_state.training_scenarios):
        # breakpoint()
        if training_scenario is None: continue
        with tabs[idx]:
            st.markdown(training_scenario)
    
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                key=f"feedback_{st.session_state.feedback_key}_{idx}",
                optional_text_label="Please provide some more information",
                max_text_length=500,
                on_submit=add_feedback,
                kwargs={"option_idx": idx}
            )

            # Add spacing after the feedback buttons and before displaying the messages
            st.markdown("<br>", unsafe_allow_html=True)

# Download generated scenarios and feedback as CSV files
if st.session_state.training_scenarios[0] is not None:
    export_rows = []
    source_query = st.session_state.get("last_query", "")
    for idx, scenario in enumerate(st.session_state.training_scenarios):
        if scenario:
            export_rows.append({"Query": source_query, "Option": idx, "Scenario": scenario})

    if export_rows:
        scenarios_csv = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download scenarios as CSV",
            data=scenarios_csv,
            file_name="training_scenarios.csv",
            mime="text/csv",
        )

# Option to download the feedback data as a CSV file
st.download_button(
    label="Download feedback data as CSV",
    data=st.session_state.data.to_csv(index=False).encode('utf-8'),
    file_name='feedback_data.csv',
    mime='text/csv',
)