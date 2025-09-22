import json5
import re

class ToolLibrary:
    def __init__(self, tools):
        self.tools = tools

    def get_tool(self, agent_output, environment_kwargs, valid_tools):
        """
        Processes agent output to extract and execute valid tools.

        This function processes the output of an agent, identifies any calls to valid
        tools, and executes them using the provided environment and tool parameters.
        Each tool call is formatted as `<tool_name>{ ... }</tool_name>` in the agent's
        output. If a matching tool is found, its execution is returned as a callable
        function, with parameters merged from the environment and the agent's output.

        Parameters:
            agent_output: str
                The textual output from the agent containing potential tool calls.
            environment_kwargs: dict
                Dictionary of environment parameters to be used or overridden by tool
                parameters.
            valid_tools: Union[str, List[str]]
                A list of valid tool names or a single tool name to be identified and
                executed.

        Returns:
            Callable:
                A callable function to execute the identified tool with its merged
                parameters, or None if no valid tool call is detected.

        Raises:
            json5.JSON5DecodeError:
                If the agent output contains a tool call with invalid JSON content.
        """

        if isinstance(valid_tools, str):
            valid_tools = [valid_tools]

        agent_output = agent_output.upper()

        for vt in valid_tools:
            match = re.search(r"(?<=<" + vt + ">).*(?=</" + vt + ">)", agent_output)  # TODO first search only; do we need
            llm_kwargs = {}

            if match is not None:
                try:
                    llm_kwargs = json5.loads(match.group(0))
                    print(llm_kwargs)
                except:
                    print(f"Tool " + vt + " called by agent but failed parsing JSON: {e}\nJSON: " + match.group(0))

                kwargs = environment_kwargs | llm_kwargs # overwrite environment

                return lambda: self.tools[vt](**kwargs)
                # assume the tool produces a seq of actions for now; future tools might not TODO

        # else we didn't detect a tool call
        return None