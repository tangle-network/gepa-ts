#!/usr/bin/env python3
"""
Python DSPy Executor Service
Executes evolved DSPy programs and returns results via JSON
"""

import sys
import json
import traceback
import dspy
from typing import Dict, Any, List
import os

def execute_dspy_program(
    program_code: str,
    examples: List[Dict[str, Any]],
    task_lm_config: Dict[str, Any],
    metric_fn_code: str = None
) -> Dict[str, Any]:
    """Execute a DSPy program with given examples and configuration."""
    try:
        # Set up the language model
        lm = dspy.LM(
            model=task_lm_config.get('model', 'openai/gpt-4o-mini'),
            api_key=task_lm_config.get('apiKey'),
            max_tokens=task_lm_config.get('maxTokens', 2000),
            temperature=task_lm_config.get('temperature', 0.7)
        )
        
        # Compile and execute the program code
        context = {}
        compile(program_code, "<string>", "exec")
        exec(program_code, context)
        
        # Get the program from context
        program = context.get("program")
        if program is None:
            return {
                "success": False,
                "error": "No 'program' object defined in code"
            }
        
        if not isinstance(program, dspy.Module):
            return {
                "success": False,
                "error": f"'program' is {type(program)}, not dspy.Module"
            }
        
        # Set the language model
        program.set_lm(lm)
        
        # Execute metric function if provided
        metric_fn = None
        if metric_fn_code:
            metric_context = {}
            exec(metric_fn_code, metric_context)
            metric_fn = metric_context.get("metric_fn")
        
        # Run predictions on examples
        results = []
        scores = []
        traces = []
        
        for example in examples:
            try:
                # Convert example to DSPy Example
                dspy_example = dspy.Example(**example.get('inputs', example))
                
                # Run prediction with trace capture
                with dspy.settings.context(trace=[]):
                    prediction = program(**dspy_example.inputs())
                    
                    # Capture trace
                    trace = dspy.settings.trace
                    
                    # Calculate score if metric provided
                    score = 0
                    feedback = None
                    if metric_fn:
                        metric_result = metric_fn(dspy_example, prediction)
                        if hasattr(metric_result, 'score'):
                            score = metric_result.score
                            feedback = getattr(metric_result, 'feedback', None)
                        else:
                            score = metric_result
                    
                    results.append({
                        "outputs": dict(prediction),
                        "score": score,
                        "feedback": feedback
                    })
                    
                    scores.append(score)
                    
                    # Format trace for serialization
                    trace_data = []
                    for t in trace:
                        if len(t) >= 3:
                            module, inputs, outputs = t[:3]
                            trace_data.append({
                                "module": str(module),
                                "inputs": {k: str(v) for k, v in inputs.items()},
                                "outputs": {k: str(v) for k, v in dict(outputs).items()} if outputs else {}
                            })
                    traces.append(trace_data)
                    
            except Exception as e:
                results.append({
                    "outputs": {},
                    "score": 0,
                    "error": str(e)
                })
                scores.append(0)
                traces.append([])
        
        return {
            "success": True,
            "results": results,
            "scores": scores,
            "traces": traces,
            "average_score": sum(scores) / len(scores) if scores else 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def main():
    """Main entry point for subprocess execution."""
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Execute the program
        result = execute_dspy_program(
            program_code=input_data['programCode'],
            examples=input_data['examples'],
            task_lm_config=input_data['taskLmConfig'],
            metric_fn_code=input_data.get('metricFnCode')
        )
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()