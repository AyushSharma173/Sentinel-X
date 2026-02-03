"""ReAct Agent Loop for clinical context investigation.

This module implements a simple ReAct (Reason+Act) loop that allows MedGemma
to dynamically investigate FHIR patient data based on imaging findings.

The loop follows this pattern:
1. Model sees imaging finding
2. Model generates THOUGHT + TOOL_CALL
3. Tool is executed against FHIR bundle
4. OBSERVATION is fed back to model
5. Repeat until FINAL_ASSESSMENT or max iterations
"""

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import (
    AGENT_MAX_ITERATIONS,
    AGENT_MAX_TOKENS_PER_TURN,
    AGENT_TOOL_CALL_TEMPERATURE,
    LOG_FULL_PROMPTS,
    LOG_FULL_RESPONSES,
    RISK_ADJUSTMENT_DECREASE,
    RISK_ADJUSTMENT_INCREASE,
    RISK_ADJUSTMENT_NONE,
)
from .json_repair import extract_final_assessment, extract_tool_call
from .logging import get_agent_trace_logger
from .prompts import build_agent_system_prompt, build_agent_user_prompt
from .state import AgentMessage, AgentState, create_initial_state
from .tools import get_all_tools, get_tool, get_tool_descriptions

logger = logging.getLogger(__name__)


class ReActAgentLoop:
    """ReAct agent loop for clinical context investigation.

    This class implements a simple while loop (not heavy frameworks like LangGraph)
    that allows the model to reason about imaging findings and query FHIR data.
    """

    def __init__(
        self,
        model: AutoModelForImageTextToText,
        processor: AutoProcessor,
        fhir_bundle: Dict[str, Any],
        patient_id: str,
        max_iterations: int = AGENT_MAX_ITERATIONS,
    ):
        """Initialize the agent loop.

        Args:
            model: The loaded MedGemma model
            processor: The model processor
            fhir_bundle: FHIR Bundle containing patient data
            patient_id: Patient identifier
            max_iterations: Maximum reasoning iterations
        """
        self.model = model
        self.processor = processor
        self.fhir_bundle = fhir_bundle
        self.patient_id = patient_id
        self.max_iterations = max_iterations

        # Build system prompt with tool descriptions
        tool_descriptions = get_tool_descriptions()
        self.system_prompt = build_agent_system_prompt(tool_descriptions, max_iterations)

    def _build_messages(self, state: AgentState) -> List[Dict[str, Any]]:
        """Build the message list for the model.

        Args:
            state: Current agent state

        Returns:
            List of messages in chat format
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

        # Add conversation history from state
        for msg in state["messages"]:
            messages.append(
                {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                }
            )

        return messages

    def _generate_response(self, state: AgentState) -> str:
        """Generate a model response.

        Args:
            state: Current agent state

        Returns:
            Model response text
        """
        messages = self._build_messages(state)

        # Apply chat template
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize (no images for agent mode - text only)
        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Generate with deterministic sampling for tool calls
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=AGENT_MAX_TOKENS_PER_TURN,
                do_sample=AGENT_TOOL_CALL_TEMPERATURE > 0,
                temperature=AGENT_TOOL_CALL_TEMPERATURE if AGENT_TOOL_CALL_TEMPERATURE > 0 else None,
            )

        # Decode response
        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )

        return response

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call against the FHIR bundle.

        Args:
            tool_call: Dict with 'tool' and 'arguments' keys

        Returns:
            Tool result dictionary
        """
        tool_name = tool_call["tool"]
        arguments = tool_call.get("arguments", {})

        # Get the tool function
        tool_func = get_tool(tool_name)
        if tool_func is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            # All tools take fhir_bundle as first argument
            result = tool_func(self.fhir_bundle, **arguments)
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return {"error": str(e)}

    def _format_observation(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format a tool result as an observation message.

        Args:
            tool_name: Name of the tool that was called
            result: Tool result dictionary

        Returns:
            Formatted observation string
        """
        import json

        result_json = json.dumps(result, indent=2, default=str)
        return f"OBSERVATION ({tool_name}):\n{result_json}"

    def _extract_and_apply_final_assessment(
        self, response: str, state: AgentState
    ) -> None:
        """Extract final assessment from response and update state.

        Args:
            response: Model response containing FINAL_ASSESSMENT
            state: Agent state to update
        """
        extracted = extract_final_assessment(response)
        if not extracted:
            logger.warning("Could not parse FINAL_ASSESSMENT")
            return

        state["final_assessment"] = extracted.get("assessment")

        # Map risk adjustment to numeric value
        risk_adj = extracted.get("risk_adjustment", "NONE")
        state["risk_adjustment"] = risk_adj

        # Extract critical findings
        state["critical_findings"] = extracted.get("critical_findings", [])

    def run(self, visual_findings: str) -> AgentState:
        """Run the ReAct agent loop.

        Args:
            visual_findings: Visual findings from MedGemma CT analysis

        Returns:
            Final agent state with assessment and risk adjustment
        """
        logger.info(f"Starting agent loop for patient: {self.patient_id}")
        trace_logger = get_agent_trace_logger()
        loop_start_time = time.time()

        # Initialize state
        state = create_initial_state(
            patient_id=self.patient_id,
            visual_findings=visual_findings,
            fhir_bundle=self.fhir_bundle,
            max_iterations=self.max_iterations,
        )

        # Add initial user message
        initial_prompt = build_agent_user_prompt(visual_findings)
        state["messages"].append(AgentMessage(role="user", content=initial_prompt))

        # Main agent loop
        while not state["should_stop"] and state["iteration"] < self.max_iterations:
            state["iteration"] += 1
            logger.info(f"Agent iteration {state['iteration']}/{self.max_iterations}")

            # Log iteration start
            trace_logger.log_iteration_start(
                patient_id=self.patient_id,
                iteration=state["iteration"],
                max_iterations=self.max_iterations,
                message_count=len(state["messages"]),
            )

            try:
                # Generate model response with timing
                gen_start_time = time.time()

                # Log prompt if enabled
                if LOG_FULL_PROMPTS:
                    messages = self._build_messages(state)
                    prompt_text = str(messages)  # Simplified for logging
                    trace_logger.log_prompt_sent(
                        patient_id=self.patient_id,
                        iteration=state["iteration"],
                        prompt=prompt_text,
                    )

                response = self._generate_response(state)
                gen_duration_ms = int((time.time() - gen_start_time) * 1000)

                logger.debug(f"Agent response: {response[:500]}...")

                # Log response received
                if LOG_FULL_RESPONSES:
                    trace_logger.log_response_received(
                        patient_id=self.patient_id,
                        iteration=state["iteration"],
                        response=response,
                        duration_ms=gen_duration_ms,
                    )

                # Add response to history
                state["messages"].append(
                    AgentMessage(role="assistant", content=response)
                )

                # Check for FINAL_ASSESSMENT (stop condition)
                if "FINAL_ASSESSMENT:" in response:
                    logger.info("Agent provided final assessment")
                    self._extract_and_apply_final_assessment(response, state)

                    # Log final assessment extraction
                    trace_logger.log_final_assessment_extracted(
                        patient_id=self.patient_id,
                        iteration=state["iteration"],
                        assessment=state.get("final_assessment", ""),
                        risk_adjustment=state.get("risk_adjustment", "NONE"),
                        critical_findings=state.get("critical_findings", []),
                    )

                    state["should_stop"] = True
                    continue

                # Try to extract tool call
                tool_call = extract_tool_call(response)
                if tool_call is None:
                    logger.warning("No tool call found in response, prompting for action")

                    # Log failed extraction
                    trace_logger.log_tool_call_failed(
                        patient_id=self.patient_id,
                        iteration=state["iteration"],
                        raw_text=response[:1000],
                        error="No TOOL_CALL pattern found",
                    )

                    # Add a nudge to either call a tool or conclude
                    state["messages"].append(
                        AgentMessage(
                            role="user",
                            content="Please either make a TOOL_CALL to investigate further, or provide your FINAL_ASSESSMENT if you have enough information.",
                        )
                    )
                    continue

                # Record tool call
                tool_name = tool_call["tool"]
                tool_args = tool_call.get("arguments", {})

                # Check for duplicate tool calls
                import json
                tool_signature = f"{tool_name}({json.dumps(tool_args, sort_keys=True)})"

                if "tool_signatures" not in state:
                    state["tool_signatures"] = set()

                if tool_signature in state["tool_signatures"]:
                    logger.warning(f"Duplicate tool call detected: {tool_signature}")

                    # Add warning to conversation
                    duplicate_msg = (
                        f"⚠️ You already called {tool_name} with these exact arguments "
                        f"in a previous iteration. The tool returned the same result as before. "
                        f"Consider trying a different approach or providing your final assessment "
                        f"if you have enough information."
                    )
                    state["messages"].append(AgentMessage(role="user", content=duplicate_msg))

                    # Log duplicate detection
                    trace_logger.log_tool_call_failed(
                        patient_id=self.patient_id,
                        iteration=state["iteration"],
                        raw_text=tool_signature,
                        error="Duplicate tool call detected",
                    )
                    continue

                # Track signature
                state["tool_signatures"].add(tool_signature)
                state["tools_used"].append(tool_name)
                state["tool_calls"].append(tool_call)

                # Log successful tool call extraction
                trace_logger.log_tool_call_extracted(
                    patient_id=self.patient_id,
                    iteration=state["iteration"],
                    tool_name=tool_name,
                    tool_args=tool_args,
                    raw_text=response[:500],
                )

                logger.info(f"Executing tool: {tool_name}")

                # Execute tool with timing
                tool_start_time = time.time()
                result = self._execute_tool(tool_call)
                tool_duration_ms = int((time.time() - tool_start_time) * 1000)

                state["tool_results"].append(result)

                # Log tool execution
                trace_logger.log_tool_execution(
                    patient_id=self.patient_id,
                    iteration=state["iteration"],
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=result,
                    duration_ms=tool_duration_ms,
                )

                # Format and add observation
                observation = self._format_observation(tool_name, result)
                state["messages"].append(AgentMessage(role="user", content=observation))

                # Log observation added
                trace_logger.log_observation_added(
                    patient_id=self.patient_id,
                    iteration=state["iteration"],
                    observation=observation,
                )

            except Exception as e:
                logger.error(f"Agent loop error: {e}", exc_info=True)
                state["errors"].append(str(e))
                # Continue to try to recover

        # If we hit max iterations without final assessment, force one
        if not state["should_stop"]:
            logger.warning("Max iterations reached, forcing conclusion")

            # Check if last message was already from user (observation)
            # If so, append to it rather than creating a new user message
            # to avoid role alternation violation
            force_conclusion_text = (
                "You have reached the maximum number of iterations. "
                "Please provide your FINAL_ASSESSMENT now based on the information gathered."
            )

            if state["messages"] and state["messages"][-1]["role"] == "user":
                # Append to existing user message to avoid role alternation violation
                state["messages"][-1]["content"] += f"\n\n{force_conclusion_text}"
            else:
                state["messages"].append(
                    AgentMessage(
                        role="user",
                        content=force_conclusion_text,
                    )
                )

            try:
                response = self._generate_response(state)
                state["messages"].append(
                    AgentMessage(role="assistant", content=response)
                )
                if "FINAL_ASSESSMENT:" in response:
                    self._extract_and_apply_final_assessment(response, state)

                    # Log final assessment extraction
                    trace_logger.log_final_assessment_extracted(
                        patient_id=self.patient_id,
                        iteration=state["iteration"],
                        assessment=state.get("final_assessment", ""),
                        risk_adjustment=state.get("risk_adjustment", "NONE"),
                        critical_findings=state.get("critical_findings", []),
                    )
            except Exception as e:
                logger.error(f"Failed to get final assessment: {e}")
                state["errors"].append(f"Failed to get final assessment: {e}")

        # Calculate total duration
        loop_duration_ms = int((time.time() - loop_start_time) * 1000)

        logger.info(
            f"Agent loop complete: {state['iteration']} iterations, "
            f"{len(state['tools_used'])} tools used, "
            f"risk_adjustment={state['risk_adjustment']}"
        )

        # Log agent completion
        trace_logger.log_agent_complete(
            patient_id=self.patient_id,
            iterations=state["iteration"],
            tools_used=state["tools_used"],
            errors=state["errors"],
            duration_ms=loop_duration_ms,
            risk_adjustment=state.get("risk_adjustment"),
        )

        return state


def get_risk_adjustment_value(adjustment: Optional[str]) -> int:
    """Convert risk adjustment string to numeric value.

    Args:
        adjustment: "INCREASE", "DECREASE", or "NONE"/None

    Returns:
        Numeric adjustment for priority level
    """
    if adjustment == "INCREASE":
        return RISK_ADJUSTMENT_INCREASE  # -1 = more urgent
    elif adjustment == "DECREASE":
        return RISK_ADJUSTMENT_DECREASE  # +1 = less urgent
    return RISK_ADJUSTMENT_NONE
