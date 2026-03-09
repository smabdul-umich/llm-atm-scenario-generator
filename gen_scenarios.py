# Generates training scenarios based on prompts in an input CSV file, using Azure OpenAI. Writes generated scenarios back to CSV.
# Expects input CSV with columns: "Prompt", "Scenario", "1 if LLM generated, 0 otherwise", "Few-shot to skip"
# To run: python gen_scenarios.py --input input_scenarios.csv --output output_scenarios.csv

import argparse
import os
import re
import sys
import pandas as pd
from dotenv import load_dotenv
from prompts import instruction_prompt, query_template, few_shot_examples, semantic_parsing_instruction
from langchain_openai import AzureChatOpenAI

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment file for secrets.
if not load_dotenv('env'):
    print('Unable to load .env file.')
    sys.exit(1)

# Load LLM with Azure OpenAI credentials from .env file
llm = AzureChatOpenAI(
    deployment_name=os.environ['model'],  # e.g. gpt-35-turbo
    openai_api_version=os.environ['API_VERSION'],  # e.g. 2023-05-15
    openai_api_key=os.environ['OPENAI_API_KEY'],  # secret
    azure_endpoint=os.environ['openai_api_base'],  # a URL
    openai_organization=os.environ['OPENAI_organization']  
)

# optional weather context to augment scenario generation
# wx_context = """
# Here is weather data you should use in the training script you generate for EWR. In the training script you generate, maintain the same format as in the example scripts, and make sure the script you generate is complete:
# Locid	Date	Local_hour	Wind_Dir	Temp (F)	Wind_Speed	Ceiling	Visibility	Airport_WX	Nearby_TS	Enroute_TS
# EWR	5/30/18	0	100	73	4	999	10		0	0
# EWR	5/30/18	1	100	71	8	999	10		0	0
# EWR	5/30/18	2	100	68	6	999	10		0	0
# EWR	5/30/18	3	60	62	4	4	1.5		0	0
# EWR	5/30/18	4	60	64	4	3	2	BR	0	0
# EWR	5/30/18	5	70	64	4	2	2	BR	0	0
# EWR	5/30/18	6	70	64	4	2	2	BR	0	0
# EWR	5/30/18	7	80	64	4	3	2	#NAME?	0	0
# EWR	5/30/18	8	90	66	4	5	2	#NAME?	0	0
# EWR	5/30/18	9	110	66	5	6	2		0	0
# EWR	5/30/18	10	VRB	69	3	8	10		0	0
# EWR	5/30/18	11	80	69	8	11	10		0	0
# EWR	5/30/18	12	80	69	7	15	10		0	0
# EWR	5/30/18	13	140	69	7	16	10		0	0
# EWR	5/30/18	14	180	68	8	13	10		0	0
# EWR	5/30/18	15	180	68	8	13	10		0	0
# EWR	5/30/18	16	190	66	7	250	10		0	0
# EWR	5/30/18	17	190	66	7	250	10		0	0
# EWR	5/30/18	18	140	59	8	6	10		0	0
# EWR	5/30/18	19	120	59	7	3	7		0	0
# EWR	5/30/18	20	120	59	7	3	7		0	0
# EWR	5/30/18	21	0	60	0	4	6	#NAME?	0	0
# EWR	5/30/18	22	0	62	0	7	10		0	0
# EWR	5/30/18	23	70	62	4	32	9		0	0
# """


def strip_markdown(text: str) -> str:
    """
    Remove markdown formatting from text for plain-text CSV output.
    """
    # Remove headers (# ## ### etc.)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove bold (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    # Remove italic (*text* or _text_) - be careful not to affect underscores in words
    text = re.sub(r'(?<!\w)\*([^*]+)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', text)
    # Remove inline code (`text`)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove horizontal rules (---, ***, ___)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Remove bullet points (- or * at start of line) but keep the content
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Remove numbered list markers
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Remove blockquotes (>)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    # Remove extra blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def generate_scenario(training_prompt: str, skip_few_shot: int = None):
    """
    Generate a single training scenario.
    
    Args:
        training_prompt: The prompt to generate the scenario from
        skip_few_shot: 1-indexed few-shot example to omit (e.g., 1 = skip first example)
    """
    # Filter few-shot examples if needed
    examples_to_use = few_shot_examples.copy()
    if skip_few_shot is not None and 1 <= skip_few_shot <= len(few_shot_examples):
        examples_to_use = [ex for i, ex in enumerate(few_shot_examples) if i != (skip_few_shot - 1)]
    
    prompt = instruction_prompt + "\n\n".join(
        query_template.format(nl_query=ex[0], sql_query=ex[1]) for ex in examples_to_use
    )
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", prompt + "\n\n" + semantic_parsing_instruction + f" {training_prompt}"),
    ]

    response = llm.invoke(messages)
    response_text = response.content
    parsed_response = re.search(r'```([\s\S]*?)```', response_text)
    parsed_response = parsed_response.group(1).strip() if parsed_response else response_text.strip()
    
    return parsed_response


def process_input_csv(input_path: str = "input_scenarios.csv", output_path: str = None):
    """
    Read prompts from input CSV, generate scenarios, and write back.
    
    CSV columns:
        - Prompt: The scenario prompt
        - Scenario: Generated scenario text (skip if already filled)
        - 1 if LLM generated, 0 otherwise: Set to 1 for LLM-generated
        - Few-shot to skip: 1-indexed few-shot example to omit
    """
    if output_path is None:
        output_path = input_path  # Overwrite input file
    
    # Read CSV
    df = pd.read_csv(input_path)
    
    # Validate columns
    required_cols = ["Prompt", "Scenario", "1 if LLM generated, 0 otherwise", "Few-shot to skip"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing required column: '{col}'")
            sys.exit(1)
    
    generated_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        # Skip rows that already have a scenario
        if pd.notna(row["Scenario"]) and str(row["Scenario"]).strip():
            skipped_count += 1
            print(f"⏭️  Row {idx + 1}: Skipping (scenario already exists)")
            continue
        
        prompt = row["Prompt"]
        if pd.isna(prompt) or not str(prompt).strip():
            print(f"⚠️  Row {idx + 1}: Skipping (empty prompt)")
            continue
        
        # Get few-shot to skip (convert to int, handle NaN)
        skip_few_shot = None
        if pd.notna(row["Few-shot to skip"]):
            try:
                skip_few_shot = int(row["Few-shot to skip"])
            except ValueError:
                pass
        
        print(f"🔄 Row {idx + 1}: Generating scenario (skip few-shot: {skip_few_shot or 'none'})...")
        
        try:
            scenario = generate_scenario(prompt, skip_few_shot)
            # Strip markdown for plain-text CSV output
            scenario = strip_markdown(scenario)
            df.at[idx, "Scenario"] = scenario
            df.at[idx, "1 if LLM generated, 0 otherwise"] = 1
            generated_count += 1
            print(f"✅ Row {idx + 1}: Generated successfully")
        except Exception as e:
            print(f"❌ Row {idx + 1}: Error generating scenario: {e}")
    
    # Write back to CSV with UTF-8 BOM encoding for Excel compatibility
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n{'='*50}")
    print(f"✅ Done! Generated: {generated_count}, Skipped: {skipped_count}")
    print(f"📄 Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training scenarios from input CSV.")
    parser.add_argument("--input", default="input_scenarios.csv", help="Input CSV file path")
    parser.add_argument("--output", default=None, help="Output CSV file path (default: overwrite input)")
    args = parser.parse_args()

    process_input_csv(args.input, args.output)


if __name__ == "__main__":
    main()