#!/usr/bin/env python3
import random
import os
import weave
import asyncio
import tempfile
import shutil

# Training configuration
import art
from art.local import LocalBackend
from art.utils import iterate_dataset
from art.langgraph import wrap_rollout
from art.rewards import ruler_score_group

from tasks.email.scenarios import load_training_scenarios
from tasks.email.rollout import rollout
from tasks.email.model import EmailScenario

from dotenv import load_dotenv

def cleanup_art_directories():
    """Clean up any existing .art directories"""
    for item in os.listdir("."):
        if item.startswith(".art"):
            if os.path.isdir(item):
                print(f"Removing old directory: {item}")
                shutil.rmtree(item, ignore_errors=True)

async def run_training():
    load_dotenv()

    # 환경 변수 체크
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for RULER functionality.")

    training_config = {
        "groups_per_step": 2,
        "num_epochs": 20,
        "rollouts_per_group": 4,
        "learning_rate": 1e-5,
        "max_steps": 20,
    }
    SCENARIO_DATASET_REPO_ID = "corbt/enron_emails_sample_questions"


    # Clean up before starting
    cleanup_art_directories()
    
    ####################  1. Load training scenarios ################### 
    training_scenarios = load_training_scenarios(
        split="train", limit=50, max_messages=1, shuffle=True, seed=42,
        SCENARIO_DATASET_REPO_ID=SCENARIO_DATASET_REPO_ID
    
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

    ###################  2. Creating a Model ################### 
    random.seed(42)

    # Use a simpler backend configuration
    with tempfile.TemporaryDirectory(prefix="art_") as temp_dir:
        
        # Declare the model
        model = art.TrainableModel(
            name="email-agent-langgraph-test",
            project="email-search-agent-test",
            base_model="Qwen/Qwen2.5-7B-Instruct",
        )

        # Simplified config for testing
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=4096,  # Reduced
            ),
            engine_args=art.dev.EngineArgs(
                enforce_eager=True,
                gpu_memory_utilization=0.7,  # Reduced
            ),
        )

        # Initialize the server with temporary directory
        backend = LocalBackend(
            in_process=True,  # Keep in process but use temp directory
            path=temp_dir,
        )

        try:
            print("Registering model with backend...")
            await model.register(backend)
            print("Model registered successfully!")

            # Weave 초기화 (optional)
            if os.getenv("WANDB_API_KEY", ""):
                weave.init(model.project, settings={"print_call_link": False})

            ################### 3. Training Loop ################### 
            training_iterator = iterate_dataset(
                training_scenarios,
                groups_per_step=training_config["groups_per_step"],
                num_epochs=training_config["num_epochs"],
                initial_step=await model.get_step(),
            )

            for batch in training_iterator:
                print(
                    f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}"
                )
                print(f"Batch contains {len(batch.items)} scenarios")

                # Create trajectory groups for this batch
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
                
                print(f"Created {len(groups)} trajectory groups")
                
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
                    if judged_group:
                        judged_groups.append(judged_group)

                if judged_groups:
                    await model.delete_checkpoints()
                    await model.train(
                        judged_groups,
                        config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
                        _config={"logprob_calculation_chunk_size": 16},
                    )
                    print(f"Completed training step {batch.step}")
                else:
                    print(f"No judged groups for step {batch.step}, skipping training")

                # Stop after max_steps
                if batch.step >= training_config["max_steps"]:
                    break

            ################### 4. Simple Test ################### 
            print("Testing the trained model...\n")

            test_scenario = training_scenarios[0]
            print(f"Test scenario: {test_scenario.question}")
            
            test_email_scenario = EmailScenario(step=0, scenario=test_scenario)
            result_trajectory = await wrap_rollout(model, rollout)(model, test_email_scenario)

            if result_trajectory.final_answer:
                print(f"Agent's Answer: {result_trajectory.final_answer.answer}")
                print(f"Expected Answer: {test_scenario.answer}")
            else:
                print("No final answer provided")

            print("\n🎉 Training completed!")
            
        except Exception as e:
            print(f"Error: {e}")
            raise
        finally:
            # Cleanup happens automatically with temporary directory
            pass

def main():
    try:
        asyncio.run(run_training())
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())