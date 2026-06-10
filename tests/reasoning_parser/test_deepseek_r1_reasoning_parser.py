import pytest

from lmbench.reasoning_parser.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

SIMPLE_REASONING = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning_content": "This is a reasoning section",
    "content": "",
}
NO_REASONING = {
    "output": "This is a reasoning section",
    "reasoning_content": None,
    "content": "This is a reasoning section",
}
MULTIPLE_LINES = {
    "output": "<think>This\nThat</think>This is the rest\nThat",
    "reasoning_content": "This\nThat",
    "content": "This is the rest\nThat",
}
SHORTEST_REASONING = {
    "output": "<think></think>This is the rest",
    "reasoning_content": "",
    "content": "This is the rest",
}
STRIP_REASONING = {
    "output": "<think>  This is reasoning with whitespaces for stripping  </think>This is the rest",
    "reasoning_content": "This is reasoning with whitespaces for stripping",
    "content": "This is the rest",
}

TEST_CASES = [SIMPLE_REASONING, COMPLETE_REASONING, NO_REASONING, MULTIPLE_LINES, SHORTEST_REASONING, STRIP_REASONING]


@pytest.fixture
def parser():
    return DeepSeekR1ReasoningParser()


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_extract_reasoning_content(parser, test_case):
    llm_output = parser.extract_reasoning_content(test_case["output"])
    if test_case["reasoning_content"] is None:
        assert isinstance(llm_output, str)
    else:
        assert llm_output.reasoning_output == test_case["reasoning_content"]
    assert llm_output == test_case["content"]
