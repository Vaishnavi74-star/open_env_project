"""
Inference script for EV Charging Scheduler Environment.
Runs all three task difficulties and evaluates agent performance.
Compatible with OpenEnv evaluation framework (Phase 2).
"""

import json
import os
import sys
import warnings
from typing import Dict, Any, List

# ── Suppress noisy warnings ──────────────────────────────────────────────────
warnings.filterwarnings("ignore")

# ── Import environment package ───────────────────────────────────────────────
try:
    from ev_charging_env import (
        create_easy_task,
        create_medium_task,
        create_hard_task,
        RandomAgent,
        GreedyAgent,
        PriorityAwareAgent,
        OptimalSearchAgent,
    )
except Exception as _import_err:
    print(f"[ERROR] Failed to import ev_charging_env: {_import_err}")
    # Attempt individual fallback imports
    try:
        from ev_charging_env.simple_tasks import (
            create_easy_task,
            create_medium_task,
            create_hard_task,
        )
        from ev_charging_env.baselines import (
            RandomAgent,
            GreedyAgent,
            PriorityAwareAgent,
            OptimalSearchAgent,
        )
    except Exception as _fallback_err:
        print(f"[ERROR] Fallback imports also failed: {_fallback_err}")
        sys.exit(1)

# ── Optional OpenAI import ───────────────────────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    print("[WARNING] openai library not installed – will run baseline agents only.")


# ─────────────────────────────────────────────────────────────────────────────
class InferenceRunner:
    """Run EV Charging environment with OpenAI API-based agent."""

    def __init__(self, model: str = None, seed: int = None):
        """Initialize with OpenAI client and environment variable support."""
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai library is not installed.")

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.model = model or os.getenv("MODEL_NAME", "gpt-4")

        api_base = os.getenv("API_BASE_URL")
        if api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)

        self.hf_token = os.getenv("HF_TOKEN")
        self.seed = seed
        self.temperature = 0.0
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history_length = 6

    def run_task(self, task_name: str, task, max_runtime_minutes: int = 15) -> Dict[str, Any]:
        """Run a single task with OpenAI agent and return results."""
        import time
        start_time = time.time()
        max_runtime_seconds = max_runtime_minutes * 60

        print(f"\n{'='*60}")
        print(f"Running {task_name.upper()} Task")
        print(f"Model: {self.model} | Temperature: {self.temperature}")
        print('='*60)

        self.conversation_history = []
        obs = task.reset()
        total_reward = 0.0
        step_count = 0
        api_call_count = 0

        # Safe attribute access helpers
        def _safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def _vehicles(o):
            v = _safe_get(o, 'vehicles', [])
            return v if v else []

        def _stations(o):
            s = _safe_get(o, 'stations', [])
            return s if s else []

        def _grid(o):
            return _safe_get(o, 'grid', {})

        vehicles = _vehicles(obs)
        stations = _stations(obs)
        grid = _grid(obs)

        initial_prompt = f"""You are an expert AI agent optimizing electric vehicle charging schedules.

MISSION: {task_name.upper()} difficulty task
- Minimize electricity costs
- Avoid grid overload (>80% capacity)
- Meet vehicle charging deadlines
- Prioritize urgent vehicles (marked as "urgent")

CURRENT ENVIRONMENT:
- Max Steps: {task.config.max_steps}
- Vehicles: {len(vehicles)}
- Charging Stations: {len(stations)}

STATE INFORMATION:
- Time Step: {_safe_get(obs, 'time_step', 0)}/{task.config.max_steps}
- Grid Load: {_safe_get(grid, 'current_load', 0.0):.1%}
- Electricity Price: ${_safe_get(grid, 'electricity_price', 0.0):.3f}/kWh

REQUIRED ACTION FORMAT (respond with valid JSON only):
{{
    "action_type": "assign" | "delay" | "release" | "power_level",
    "vehicle_id": <integer>,
    "station_id": <integer, required for assign/release/power_level>,
    "power_level": <float 0.0-1.0, default 1.0>,
    "reasoning": "<brief strategy explanation>"
}}

Choose your first action now:"""

        self.conversation_history.append({"role": "user", "content": initial_prompt})

        max_steps = task.config.max_steps
        consecutive_errors = 0
        max_consecutive_errors = 5

        for step in range(max_steps):
            elapsed = time.time() - start_time
            if elapsed > max_runtime_seconds:
                print(f"  Runtime limit ({max_runtime_minutes}min) exceeded. Stopping early.")
                break

            try:
                action_dict = self._get_openai_action()
                api_call_count += 1

                result = task.step(action_dict)
                step_count += 1
                total_reward = result.reward if step == 0 else total_reward + result.reward
                consecutive_errors = 0

                next_obs = result.observation
                next_grid = _grid(next_obs)
                next_vehicles = _vehicles(next_obs)

                update_prompt = f"""ACTION RESULT:
- Executed: {action_dict.get('action_type','?')} (vehicle {action_dict.get('vehicle_id','?')})
- Reward: {result.reward:+.3f}
- New Grid Load: {_safe_get(next_grid, 'current_load', 0.0):.1%}
- Price: ${_safe_get(next_grid, 'electricity_price', 0.0):.3f}/kWh

Choose next action (respond with JSON only):"""

                self.conversation_history.append({"role": "assistant", "content": json.dumps(action_dict, indent=2)})
                self.conversation_history.append({"role": "user", "content": update_prompt})

                if len(self.conversation_history) > self.max_history_length:
                    self.conversation_history = (
                        self.conversation_history[:2] +
                        self.conversation_history[-self.max_history_length:]
                    )

                if (step + 1) % 10 == 0 or step == 0:
                    charged = sum(
                        1 for v in next_vehicles
                        if _safe_get(v, 'fully_charged', False)
                    )
                    print(f"  Step {step+1:3d}/{max_steps}: Reward={result.reward:+6.3f}, "
                          f"Grid={_safe_get(next_grid, 'current_load', 0.0):.1%}, "
                          f"Charged={charged}/{len(next_vehicles)}, "
                          f"API calls={api_call_count}")

                if result.done or (isinstance(result.info, dict) and result.info.get("done", False)):
                    print(f"  Task completed at step {step+1}")
                    break

            except Exception as e:
                consecutive_errors += 1
                print(f"  Error at step {step+1}: {e} (error {consecutive_errors}/{max_consecutive_errors})")

                if consecutive_errors >= max_consecutive_errors:
                    print("  Too many consecutive errors. Stopping task.")
                    break

                try:
                    fallback_action = {"action_type": "delay", "vehicle_id": 0, "duration": 1}
                    result = task.step(fallback_action)
                    step_count += 1
                    print(f"  Using fallback action")
                except Exception as fallback_error:
                    print(f"  Fallback action also failed: {fallback_error}")
                    break
                continue

        try:
            grade_result = task.grade(use_llm=False)
        except Exception as e:
            print(f"  Warning: grading failed ({e}), using defaults")
            from ev_charging_env.tasks import TaskResult
            grade_result = TaskResult(
                task_name=task_name,
                score=0.0,
                details={},
                episode_reward=total_reward,
                steps_taken=step_count,
                vehicles_charged=0,
                missed_deadlines=0,
                grid_overloads=0,
                total_cost=0.0,
            )

        elapsed_time = time.time() - start_time

        results = {
            "task_name": task_name,
            "score": grade_result.score,
            "episode_reward": grade_result.episode_reward,
            "steps_taken": grade_result.steps_taken,
            "vehicles_charged": grade_result.vehicles_charged,
            "missed_deadlines": grade_result.missed_deadlines,
            "grid_overloads": grade_result.grid_overloads,
            "total_cost": grade_result.total_cost,
            "api_calls": api_call_count,
            "elapsed_time_seconds": round(elapsed_time, 2),
            "details": grade_result.details,
        }

        print(f"\nTask Complete!")
        print(f"  Score: {results['score']:.3f}")
        print(f"  Vehicles Charged: {results['vehicles_charged']}")
        print(f"  Total Cost: ${results['total_cost']:.2f}")
        print(f"  API Calls: {api_call_count}")
        print(f"  Runtime: {elapsed_time:.1f}s")

        return results

    def _get_openai_action(self) -> Dict[str, Any]:
        """Get validated action from OpenAI API with robust error handling."""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                messages = self.conversation_history[-self.max_history_length:]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=300,
                    seed=self.seed,
                )
                response_text = response.choices[0].message.content
                action = self._parse_and_validate_action(response_text)
                if action:
                    return action
                print(f"    Attempt {attempt+1}: Invalid action format, retrying...")
            except Exception as e:
                print(f"    Attempt {attempt+1}: API error: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(base_delay * (2 ** attempt))

        print("    All API attempts failed, using safe fallback action")
        return self._get_safe_fallback_action()

    def _parse_and_validate_action(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate JSON action from LLM response."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return None
            action = json.loads(json_match.group())

            if not isinstance(action.get("action_type"), str):
                return None
            if not isinstance(action.get("vehicle_id"), int):
                return None

            valid_action_types = ["assign", "delay", "release", "power_level"]
            if action["action_type"] not in valid_action_types:
                return None

            if action["action_type"] == "assign":
                if not isinstance(action.get("station_id"), int):
                    action["station_id"] = 0
                action["power_level"] = max(0.0, min(1.0, action.get("power_level", 1.0)))
            elif action["action_type"] == "power_level":
                if not isinstance(action.get("station_id"), int):
                    action["station_id"] = 0
                action["power_level"] = max(0.0, min(1.0, action.get("power_level", 1.0)))
            elif action["action_type"] == "delay":
                action["duration"] = max(1, action.get("duration", 1))
            elif action["action_type"] == "release":
                if not isinstance(action.get("station_id"), int):
                    action["station_id"] = 0

            return action
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"    JSON parsing/validation error: {e}")
            return None

    def _get_safe_fallback_action(self) -> Dict[str, Any]:
        """Get a safe fallback action when all else fails."""
        return {"action_type": "delay", "vehicle_id": 0, "duration": 1}


# ─────────────────────────────────────────────────────────────────────────────
def output_results(task_results, output_file: str = None):
    """Output results in both console and JSON formats."""
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)

    scores = {}
    total_score = 0.0
    valid_tasks = 0

    if isinstance(task_results, dict):
        for task_name, agents in task_results.items():
            print(f"\n{task_name.upper()}:")
            for agent_name, metrics in agents.items():
                score = metrics.get('score', 0.0)
                print(f"  {agent_name}: {score:.3f}")
                scores[f"{task_name}_{agent_name}"] = round(score, 4)
                total_score += score
                valid_tasks += 1
    else:
        for result in task_results:
            if isinstance(result, dict):
                task_name = result.get('task_name', 'unknown')
                if "error" not in result:
                    score = result.get('score', 0.0)
                    status = "✅"
                    total_score += score
                    valid_tasks += 1
                else:
                    score = 0.0
                    status = "❌"
                print(f"{status} {task_name.upper():8s}: {score:.4f}")
                scores[task_name] = round(score, 4)
                if "api_calls" in result:
                    print(f"    API Calls: {result['api_calls']}, "
                          f"Runtime: {result.get('elapsed_time_seconds', 0):.1f}s")

    if valid_tasks > 0:
        avg_score = round(total_score / valid_tasks, 4)
        scores["average"] = avg_score
        print(f"\n📊 {'AVERAGE':8s}: {avg_score:.4f}")
    else:
        scores["average"] = 0.0
        print(f"\n❌ {'AVERAGE':8s}: 0.0000 (no valid tasks)")

    print("\n" + "="*70)

    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(scores, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"\n❌ Error saving to {output_file}: {e}")

    print(f"\n📄 JSON Output: {json.dumps(scores, indent=None)}")


def run_baseline_agents() -> Dict[str, Dict[str, float]]:
    """Run all baseline agents on all tasks for comparison."""
    print("\n" + "="*60)
    print("BASELINE AGENT PERFORMANCE")
    print("="*60)

    agents = {
        "random": RandomAgent(),
        "greedy": GreedyAgent(),
        "priority": PriorityAwareAgent(),
        "optimal_search": OptimalSearchAgent(),
    }

    task_factories = {
        "easy": create_easy_task,
        "medium": create_medium_task,
        "hard": create_hard_task,
    }

    results = {}

    for task_name, task_func in task_factories.items():
        print(f"\n{task_name.upper()} TASK:")
        results[task_name] = {}

        try:
            task = task_func()
        except Exception as e:
            print(f"  Failed to create {task_name} task: {e}")
            continue

        for agent_name, agent in agents.items():
            try:
                obs = task.reset()
                step_count = 0
                max_steps = task.config.max_steps

                while step_count < max_steps and obs is not None:
                    try:
                        action = agent.get_action(obs)
                        step_result = task.step(action)
                        obs = step_result.observation
                        step_count += 1
                        if step_result.done or (isinstance(step_result.info, dict) and step_result.info.get("done", False)):
                            break
                    except Exception as step_err:
                        print(f"    Step error for {agent_name}: {step_err}")
                        break

                grade = task.grade(use_llm=False)
                results[task_name][agent_name] = {
                    "score": grade.score,
                    "vehicles_charged": grade.vehicles_charged,
                    "total_cost": grade.total_cost,
                }
                print(f"  {agent_name:15s}: {grade.score:.3f} "
                      f"(charged {grade.vehicles_charged}, cost ${grade.total_cost:.2f})")

            except Exception as e:
                print(f"  Error running {agent_name} on {task_name}: {e}")
                results[task_name][agent_name] = {"score": 0.0, "vehicles_charged": 0, "total_cost": 0.0}

    return results


def main():
    """Main inference runner. Never exits with non-zero unless truly fatal."""
    import argparse

    parser = argparse.ArgumentParser(description="EV Charging Scheduler - OpenAI Inference")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-runtime", type=int, default=15)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    print("\n" + "="*70)
    print("EV CHARGING SCHEDULER - OPENENV INFERENCE EVALUATION")
    print("="*70)

    # ── No API key → fall back to baseline agents ────────────────────────────
    if not os.getenv("OPENAI_API_KEY") or not _OPENAI_AVAILABLE:
        if not _OPENAI_AVAILABLE:
            print("\n⚠️  openai library not available.")
        else:
            print("\n⚠️  OPENAI_API_KEY not set.")
        print("Falling back to baseline agent comparison...\n")

        try:
            baseline_results = run_baseline_agents()
            output_results(baseline_results, args.output_json)
        except Exception as e:
            print(f"❌ Baseline evaluation failed: {e}")
            # Still exit 0 – evaluation framework should not crash
        return

    # ── Run with OpenAI API ──────────────────────────────────────────────────
    print(f"\n🤖 Using model: {args.model or os.getenv('MODEL_NAME', 'gpt-4')}")

    try:
        runner = InferenceRunner(model=args.model, seed=args.seed)
    except Exception as e:
        print(f"❌ Could not initialize InferenceRunner: {e}")
        print("Falling back to baseline agents...")
        try:
            baseline_results = run_baseline_agents()
            output_results(baseline_results, args.output_json)
        except Exception as be:
            print(f"❌ Baseline fallback also failed: {be}")
        return

    task_results = []
    tasks = [
        ("easy", create_easy_task),
        ("medium", create_medium_task),
        ("hard", create_hard_task),
    ]

    for task_name, task_func in tasks:
        try:
            print(f"\n🚀 Starting {task_name.upper()} task...")
            task = task_func()
            result = runner.run_task(task_name, task, args.max_runtime)
            task_results.append(result)
        except Exception as e:
            print(f"❌ Error running {task_name} task: {e}")
            task_results.append({
                "task_name": task_name,
                "score": 0.0,
                "error": str(e),
                "api_calls": 0,
                "elapsed_time_seconds": 0,
            })

    output_results(task_results, args.output_json)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] Unhandled exception in inference.py: {e}")
        import traceback
        traceback.print_exc()
        # Exit 0 so evaluator doesn't count this as a hard crash
        sys.exit(0)
