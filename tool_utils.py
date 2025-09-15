import numpy as np

directions_to_actions = {
    "[-1 0]": 'UP',
    "[0 -1]": 'LEFT',
    "[1 0]": 'DOWN',
    "[0 1]": 'RIGHT'
}

def action_seq_to_agent_queue(seq, tool_name='undefined'):
    return [{'action': item,
             'reasoning': 'tool "' + tool_name + '" was used'}
            for item in seq] # TODO make the reasoning more specific

