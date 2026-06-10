import os
from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from lmbench.models.data_models import LLMMessage, LLMRole
from lmbench.models.openai import AzureOpenAILLM, OpenAILLM


def test_openai(mocker: MockerFixture):
    os.environ["API_KEY"] = "test_api_key"
    llm = OpenAILLM.from_arg_string(model="gpt-4o", arg_string="base_url=test,api_key=API_KEY")
    assert isinstance(llm, OpenAILLM), "LLM should be an instance of OpenAILLM"
    mock_create = mocker.patch.object(llm.openai_client.chat.completions, "create")

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.message.content = "Response"
    mock_response.choices = [mock_message]

    mock_create.return_value = mock_response
    result = llm.invoke([LLMMessage(role=LLMRole.USER, content="Hello")])
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o", "Model should be gpt-4o"
    assert kwargs["messages"] == [{"role": "user", "content": "Hello"}], "Messages should match expected format"
    assert result == "Response", "Result should match the mock response content"


def test_azure_openai(mocker: MockerFixture):
    os.environ["API_KEY"] = "test_api_key"
    llm = AzureOpenAILLM.from_arg_string(
        model="gpt-4o", arg_string="azure_endpoint=test,api_version=test,api_key=API_KEY"
    )
    assert isinstance(llm, AzureOpenAILLM), "LLM should be an instance of AzureOpenAILLM"
    mock_create = mocker.patch.object(llm.openai_client.chat.completions, "create")

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.message.content = "Response"
    mock_response.choices = [mock_message]

    mock_create.return_value = mock_response
    result = llm.invoke([LLMMessage(role=LLMRole.USER, content="Hello")])
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o", "Model should be gpt-4o"
    assert kwargs["messages"] == [{"role": "user", "content": "Hello"}], "Messages should match expected format"
    assert result == "Response", "Result should match the mock response content"
