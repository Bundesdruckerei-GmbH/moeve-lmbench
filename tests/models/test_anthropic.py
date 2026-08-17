import os
from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from lmbench.models.anthropic import AnthropicLLM
from lmbench.models.data_models import LLMMessage, LLMOutput, LLMRole


def _mock_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def test_anthropic(mocker: MockerFixture):
    os.environ["API_KEY"] = "test_api_key"
    llm = AnthropicLLM.from_arg_string(model="claude-opus-4-7", arg_string="api_key=API_KEY")
    assert isinstance(llm, AnthropicLLM), "LLM should be an instance of AnthropicLLM"
    mock_create = mocker.patch.object(llm.anthropic_client.messages, "create")
    mock_create.return_value = _mock_text_response("Response")

    result = llm.invoke([LLMMessage(role=LLMRole.USER, content="Hello")])
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert "system" not in kwargs, "System param should be omitted when no system message is present"
    assert kwargs["max_tokens"] == llm.num_predict
    assert result == "Response"


def test_anthropic_extracts_system_message(mocker: MockerFixture):
    os.environ["API_KEY"] = "test_api_key"
    llm = AnthropicLLM.from_arg_string(model="claude-opus-4-7", arg_string="api_key=API_KEY")
    mock_create = mocker.patch.object(llm.anthropic_client.messages, "create")
    mock_create.return_value = _mock_text_response("Hi there")

    llm.invoke(
        [
            LLMMessage(role=LLMRole.SYSTEM, content="You are helpful."),
            LLMMessage(role=LLMRole.USER, content="Hello"),
        ]
    )
    kwargs = mock_create.call_args.kwargs
    assert kwargs["system"] == "You are helpful."
    assert kwargs["messages"] == [{"role": "user", "content": "Hello"}], (
        "System message should be lifted out of the messages list"
    )


def test_anthropic_tokenizer_calls_count_tokens(mocker: MockerFixture):
    os.environ["API_KEY"] = "test_api_key"
    llm = AnthropicLLM.from_arg_string(model="claude-opus-4-7", arg_string="api_key=API_KEY,tokenizer=anthropic")
    assert llm.tokenizer.name == "anthropic"

    mock_count = mocker.patch.object(llm.anthropic_client.messages, "count_tokens")
    mock_count.return_value = MagicMock(input_tokens=42)

    n = llm.tokenizer.num_tokens(
        [
            LLMMessage(role=LLMRole.SYSTEM, content="You are helpful."),
            LLMMessage(role=LLMRole.USER, content="Hello"),
        ]
    )
    mock_count.assert_called_once()
    kwargs = mock_count.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["system"] == "You are helpful."
    assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert n == 42


def test_anthropic_tokenizer_skips_empty_content(mocker: MockerFixture):
    os.environ["API_KEY"] = "test_api_key"
    llm = AnthropicLLM.from_arg_string(model="claude-opus-4-7", arg_string="api_key=API_KEY,tokenizer=anthropic")
    mock_count = mocker.patch.object(llm.anthropic_client.messages, "count_tokens")

    for content in ["", "   ", "\n\t "]:
        n = llm.tokenizer.num_tokens([LLMMessage(role=LLMRole.USER, content=content)])
        assert n == 0, f"Expected 0 tokens for empty content {content!r}"

    mock_count.assert_not_called()


def test_anthropic_returns_llm_output_with_thinking(mocker: MockerFixture):
    os.environ["API_KEY"] = "test_api_key"
    llm = AnthropicLLM.from_arg_string(model="claude-opus-4-7", arg_string="api_key=API_KEY")
    mock_create = mocker.patch.object(llm.anthropic_client.messages, "create")

    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.thinking = "step-by-step reasoning"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "final answer"
    response = MagicMock()
    response.content = [thinking_block, text_block]
    mock_create.return_value = response

    result = llm.invoke([LLMMessage(role=LLMRole.USER, content="Hello")])
    assert isinstance(result, LLMOutput)
    assert result == "final answer"
    assert result.reasoning_output == "step-by-step reasoning"
    assert result.original_output == "<think>step-by-step reasoning</think>final answer"
