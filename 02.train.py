import random



# Training configuration
import art
from art.local import LocalBackend
from art.utils import iterate_dataset
from art.langgraph import wrap_rollout
from art.rewards import ruler_score_group

from tasks.email.scenarios import load_training_scenarios
from tasks.email.rollout import rollout
from tasks.email.model import EmailScenario

training_config = {
    "groups_per_step": 2,
    "num_epochs": 20,
    "rollouts_per_group": 4,
    "learning_rate": 1e-5,
    "max_steps": 20,
}





####################  1. Load training scenarios ################### 
training_scenarios = load_training_scenarios(
    split="train", limit=50, max_messages=1, shuffle=True, seed=42
)

print("Email search environment created with full Enron dataset!")
print(
    f"Database contains the complete email dataset, loaded {len(training_scenarios)} training scenarios."
)

# print first scenario
print("\nSample scenario")
print("id:", training_scenarios[0].id)
print("question:", training_scenarios[0].question)
print("answer:", training_scenarios[0].answer)
print("message_ids:", training_scenarios[0].message_ids)
print("how_realistic:", training_scenarios[0].how_realistic)
print("inbox_address:", training_scenarios[0].inbox_address)
print("query_date:", training_scenarios[0].query_date)
print("split:", training_scenarios[0].split)



###################  2. Creating a Model ################### 
random.seed(42)

# Declare the model
model = art.TrainableModel(
    name="email-agent-langgraph-001",
    project="email-search-agent-langgraph",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)

# To run on a T4, we need to override some config defaults.
model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(
        max_seq_length=8192,
    ),
    engine_args=art.dev.EngineArgs(
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    ),
)

# Initialize the server
backend = LocalBackend(
    # Normally we don't want to run the server in-process, but for the output
    # to show up properly on Google Colab we'll enable this.
    in_process=True,
    path="./.art",
)

# Register the model with the local Backend (sets up logging, inference, and training)
await model.register(backend)




# Use iterate_dataset with real training scenarios (similar to train.py)
training_iterator = iterate_dataset(
    training_scenarios,  # Use real scenarios from Hugging Face
    groups_per_step=training_config["groups_per_step"],
    num_epochs=training_config["num_epochs"],
    initial_step=await model.get_step(),
)

for batch in training_iterator:
    print(
        f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}"
    )
    print(f"Batch contains {len(batch.items)} scenarios")

    # Create trajectory groups for this batch (similar to train.py)
    groups = []
    for scenario in batch.items:
        groups.append(
            art.TrajectoryGroup(
                (
                    wrap_rollout(model, rollout)(
                        model, EmailScenario(step=batch.step, scenario=scenario)
                    )
                    for _ in range(training_config["rollouts_per_group"])
                )
            )
        )
    print(groups[0])
    # Gather all trajectory groups
    finished_groups = await art.gather_trajectory_groups(
        groups,
        pbar_desc="gather",
        max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
    )

    judged_groups = []
    for group in finished_groups:
        # Use RULER to assign relative scores to each trajectory
        judged_group = await ruler_score_group(group, "openai/o4-mini", debug=True)
        judged_groups.append(judged_group)

    await model.delete_checkpoints()
    await model.train(
        judged_groups,
        config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
        # Lowering the logprob_calculation_chunk_size is a memory saving measure
        # to allow longer sequences (up to 8192 tokens) to be processed on a T4.
        _config={"logprob_calculation_chunk_size": 8},
    )

    print(f"Completed training step {batch.step}")

    # Stop after max_steps for demo purposes (adjust as needed)
    if batch.step >= training_config["max_steps"]:
        break
    

################### 3. Inference ################### 

#@title Loading/inference code

# Test the trained model using the rollout function
# This avoids memory issues and uses the same inference path as training

print("Testing the trained LangGraph model with a real scenario...\n")


# Use a scenario from our training set
test_scenario = training_scenarios[1]

print(f"Test scenario ID: {test_scenario.id}")
print(f"Question: {test_scenario.question}")
print(f"Expected answer: {test_scenario.answer}")
print(f"Reference message IDs: {test_scenario.message_ids}")
print(f"Inbox: {test_scenario.inbox_address}")
print(f"Query date: {test_scenario.query_date}")
print("-" * 50)

# Run the rollout function with the trained model
test_email_scenario = EmailScenario.model_validate(
    {"step": 0, "scenario": test_scenario.model_dump()}
)
result_trajectory = await wrap_rollout(model, rollout)(model, test_email_scenario)

print("LangGraph Agent's trajectory:")
print("-" * 20)

# Display the conversation
messages = result_trajectory.messages()
for i, msg in enumerate(messages):
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", [])

    if role == "system":
        print(
            f"[SYSTEM]: {content[:100]}..."
            if len(content) > 100
            else f"[SYSTEM]: {content}"
        )
    elif role == "user":
        print(f"[USER]: {content}")
    elif role == "assistant":
        if tool_calls:
            print(f"[ASSISTANT]: {tool_calls}")
        if content:
            print(f"[ASSISTANT]: {content}")
    elif role == "tool":
        tool_name = msg.get("name", "unknown_tool")
        print(
            f"[TOOL - {tool_name}]: {content[:200]}..."
            if len(content) > 200
            else f"[TOOL - {tool_name}]: {content}"
        )

    print()

print("-" * 50)
if result_trajectory.final_answer:
    print(f"Agent's Final Answer: {result_trajectory.final_answer.answer}")
    print(f"Source IDs Used: {result_trajectory.final_answer.source_ids}")
else:
    print("No final answer provided by the agent")

print(f"\nExpected Answer: {test_scenario.answer}")
print(f"Expected Source IDs: {test_scenario.message_ids}")

print("\n🎉 LangGraph email search agent testing completed!")
print(
    "The agent used LangGraph's ReAct pattern with the same inference path as during training."
)











