# Evaluates ATM training scenarios using GEval metrics and custom rubric/evaluation steps. Reads scenarios from input CSV, runs all evaluations in a single test run, and writes results to output CSV.
# Expects input CSV with same columns as in gen_scenarios.py ("Prompt", "Scenario")
# To run: python eval.py --input input_scenarios.csv --output eval_results.csv

import os
import argparse
import pandas as pd
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig

# Set openAI API key from terminal with following command:
# export OPENAI_API_KEY="your-api-key-here"


eval_model = "gpt-5.2"

# 1. Define the individual metrics based on your rubric
# We include the "Score 1" and "Score 5" definitions directly in the criteria.

instructional_metric = GEval(
    name="Instructional Guidance",
    model=eval_model,
    criteria="""Determine the level of guidance provided for the instructor and the clarity with which the 'correct' student response is defined.
    Score of 1: No guidance for the instructor; the 'correct' response is undefined.
    Score of 5: Includes explicit Instructor Questions and Evaluative Checklists with specific 'what to look for' prompts.""",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Check for the presence of specific questions for the instructor to ask.",
        "Look for evaluative checklists or 'what to look for' prompts.",
        "Verify if the 'correct' student response is clearly defined."
    ]
)

logical_flow_metric = GEval(
    name="Logical Flow & Evolution",
    model=eval_model,
    criteria="""Evaluate the chronological progression and time-dependent evolution of the scenario.
    Score of 1: Static narrative, no timeline, no role definitions, no progression.
    Score of 5: Scenario has a chronological flow (e.g., 1030Z to 1600Z) with scripted prompts and evolving characteristics like revised weather briefings, requiring the student to adapt to changing conditions.""",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Identify if there is a clear timeline (Zulu time or intervals).",
        "Check for role-specific scripted prompts (Student, Instructor, Weather Specialist).",
        "Determine if conditions (like weather) change over time requiring student adaptation."
    ]
)

technical_accuracy_metric = GEval(
    name="Technical Accuracy & Depth",
    model=eval_model,
    criteria="""Assess the use of domain-specific FAA/TFM phraseology and tools throughout the scenario.
    Score of 1: Generic descriptions (e.g., 'weather is bad'), casual non-domain-specific phraseology with no FAA/TFM software references.
    Score of 5: Relevant METAR/TAF sequences, references to TFM tools (FSM, TSD), and precise langauge / references to TFM actions like 1st Tier Ground Stops or Runway Configs.""",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Search for specific METAR/TAF sequences.",
        "Look for mentions of TFM tools like Flight Schedule Monitor (FSM) or Traffic Situation Display (TSD).",
        "Evaluate the use of specific ATC / air traffic management (ATM) terminology like 'Miles-In-Trail' or 'West Configuration'."
    ]
)

stakeholder_metric = GEval(
    name="Stakeholder Perspectives",
    model=eval_model,
    criteria="""Evaluate whether the scenario represents competing interests from various national airspace system (NAS) stakeholders and pushback.
    Score of 1: Scenario is written such that decisions can be made in a vacuum without outside input or pushback.
    Score of 5: Scenario contains scripted dialogue for multiple stakeholders (airlines, facilities) representing varying perspectives/priorities requiring a final call by the trainee.""",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Identify if there are multiple stakeholders involved (e.g., UAL, TRACON).",
        "Check for conflicting goals or 'pushback' scripted into the dialogue.",
        "Determine if the trainee is forced to negotiate or make a decision amidst these interests."
    ]
)

complexity_metric = GEval(
    name="Operational Complexity & Pathing",
    model=eval_model,
    criteria="""Assess whether the scenario itself encodes multiple viable operational decision paths and contingencies for instructor-led or trainee-led exploration.
    Score of 1: Scenario presents a single linear progression with no alternate
    branches, conditional injects, or downstream consequences.
    Score of 5: Scenario structurally encodes multiple viable decision paths
    (explicitly or implicitly), with conditional injects, divergent outcomes,
    and evolving information that forces trade-offs without prescribing a correct answer.""",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Identify whether multiple decision paths are structurally encoded (e.g., conditional injects, alternate responses, branching timelines).",
        "Check for unexpected changes or information reveals that alter downstream decisions.",
        "Determine whether the scenario informs decision-making without explicitly stating a correct choice."
    ]
)

# All metrics list
atc_metrics = [
    instructional_metric, 
    logical_flow_metric, 
    technical_accuracy_metric, 
    stakeholder_metric, 
    complexity_metric
]


def process_csv(input_path: str, output_path: str):
    """
    Read scenarios from CSV, evaluate all in a single test run, and write results to output CSV.
    """
    # Read input CSV
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    # Check if 'Scenario' column exists
    if 'Scenario' not in df.columns:
        raise ValueError("Input CSV must have a 'Scenario' column")
    
    # Initialize result columns
    metric_names = [m.name for m in atc_metrics]
    for name in metric_names:
        df[f"{name}_score"] = None
        df[f"{name}_reason"] = None
    
    # Build all test cases first
    test_cases = []
    valid_indices = []  # Track which rows have valid scenarios
    
    total_rows = len(df)
    print(f"Building test cases from {total_rows} rows...")
    
    for idx, row in df.iterrows():
        scenario = row['Scenario']
        input_prompt = row['Prompt']
        
        # Skip empty scenarios
        if pd.isna(scenario) or str(scenario).strip() == '':
            print(f"Row {idx + 1}/{total_rows}: Skipping (empty scenario)")
            continue
        
        print(f"Row {idx + 1}/{total_rows}: Adding test case...")
        
        test_case = LLMTestCase(
            input=str(input_prompt),
            actual_output=str(scenario)
        )
        test_cases.append(test_case)
        valid_indices.append(idx)
    
    if not test_cases:
        print("No valid scenarios found to evaluate.")
        return
    
    # Run evaluation on all test cases in a single test run
    print(f"\nEvaluating {len(test_cases)} test cases in a single test run...")

    async_config = AsyncConfig(
    max_concurrent=2,  # Limit concurrent evaluations
    throttle_value=5   # 5 second delay between API calls to avoid rate limits
    )

    eval_results = evaluate(
        async_config=async_config,
        test_cases=test_cases,
        metrics=atc_metrics
    )
    
    # Extract results from the evaluation results
    for i, (test_result, df_idx) in enumerate(zip(eval_results.test_results, valid_indices)):
        # Get metrics results from each test result
        for metric_result in test_result.metrics_data:
            metric_name = metric_result.name
            df.at[df_idx, f"{metric_name}_score"] = metric_result.score
            df.at[df_idx, f"{metric_name}_reason"] = metric_result.reason
        
        # Print summary for this row
        scores = [mr.score for mr in test_result.metrics_data if mr.score is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Row {df_idx + 1}: Average score: {avg_score:.2f}")
    
    # Save results
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    for name in metric_names:
        score_col = f"{name}_score"
        scores = df[score_col].dropna()
        if len(scores) > 0:
            print(f"{name}: avg={scores.mean():.2f}, min={scores.min():.2f}, max={scores.max():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ATM training scenarios from CSV")
    parser.add_argument(
        "--input", "-i",
        default="input_scenarios.csv",
        help="Path to input CSV file (default: input_scenarios.csv)"
    )
    parser.add_argument(
        "--output", "-o", 
        default="eval_results.csv",
        help="Path to output CSV file (default: eval_results.csv)"
    )
    
    args = parser.parse_args()
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print()
    
    process_csv(args.input, args.output)
