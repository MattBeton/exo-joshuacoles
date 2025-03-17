import json
import re
from typing import Any

from exo.api.chat_completion_request import ToolDefinition
from exo.api.inference_result_manager import InferenceResultChunk
from exo.helpers import DEBUG
from exo.inference.grammars import lark_grammar
from exo.tools.tool_parser import ToolParser, UnplacedToolCall


class LlamaLightweightToolParser(ToolParser):
    def is_start_of_tool_section(self, chunk: InferenceResultChunk):
        # In the new format, function calls start with a '['.
        return chunk.text.startswith('[')

    def to_grammar(self, tools: list[ToolDefinition], required: bool, parallel_tool_calling: bool) -> str:
        """
        Returns a Lark grammar string that describes the new plaintext function call format.
        The expected output should be a list of function calls, e.g.:
        
          [func_name1(param1=value1, param2=value2), func_name2(arg)]
        """
        return lark_grammar(f"""
%llguidance {{}}

start: {"fun_calls" if required else "TEXT | fun_calls"}
TEXT: /[^\\[](.|\n)*/
fun_calls: "[" fun_call ("," fun_call)* "]"
fun_call: FUNCTION_NAME "(" [parameters] ")"
FUNCTION_NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
?parameters: parameter ("," parameter)*
?parameter: PARAM_NAME "=" PARAM_VALUE   -> key_value
         | PARAM_VALUE                   -> value_only
PARAM_NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
PARAM_VALUE: /[^,()\\]]+/
        """.strip())

    def parse_complete(self, content: str, parallel_tool_calling: bool = False) -> list[UnplacedToolCall]:
        """
        Parses plaintext function call outputs in the form:
          [func_name1(param1=value1, param2=value2), func_name2(arg)]
        Returns a list of UnplacedToolCall objects where the "arguments" field is a JSON-dumped dict.
        If a parameter is provided without a key (as in func_name2(arg)), it is stored with the key "arg".
        """
        tool_calls = []
        content = content.strip()
        # Remove surrounding square brackets if they exist.
        if content.startswith('[') and content.endswith(']'):
            inner = content[1:-1].strip()
        else:
            inner = content

        # Regex pattern to match each function call.
        # It captures the function name and the content between the parentheses.
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\(\s*(.*?)\s*\)'
        matches = re.findall(pattern, inner)
        for func_name, params_str in matches:
            params = {}
            if params_str:
                # Split by commas (assumes no nested commas in parameter values).
                for part in re.split(r',\s*', params_str):
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key.strip()] = value.strip()
                    else:
                        # If there is no "=", treat the whole part as a single positional argument.
                        params["arg"] = part.strip()
            tool_calls.append(UnplacedToolCall(
                name=func_name,
                arguments=json.dumps(params)
            ))
        return tool_calls


def generate_plaintext_tool_call_schema(tools: list[ToolDefinition], parameter_key: str = "arguments") -> str:
    """
    Generate a plaintext schema representation for tool calling in the new Llama function calling format.
    
    For each tool, the format is:
      func_name(param1=type1, param2=type2, …)
    
    This function builds a signature for each tool based on its function name and its parameter types,
    then returns a comma-separated list of these signatures enclosed in square brackets.
    """
    tool_signatures = []
    for tool in tools:
        # Extract the function name.
        func_name = tool.function.name
        # Build a simple signature from the parameter definitions.
        params_schema = tool.function.parameters
        if "properties" in params_schema:
            params = []
            for param, details in params_schema["properties"].items():
                # Use the parameter type as a simple representation.
                params.append(f"{param}={details.get('type', 'any')}")
            params_str = ", ".join(params)
        else:
            params_str = ""
        tool_signatures.append(f"{func_name}({params_str})")
    # The schema is a list of function call signatures.
    return f"[{', '.join(tool_signatures)}]"