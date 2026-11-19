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


def _bad_request(message: str):
    import httpx
    import openai

    response = httpx.Response(400, request=httpx.Request("POST", "http://test"))
    return openai.BadRequestError(message, response=response, body=None)


def test_azure_openai_content_filter_rejection_raises_content_filter(mocker: MockerFixture):
    import pytest

    from lmbench.models.exceptions import ContentFilterException

    os.environ["API_KEY"] = "test_api_key"
    llm = AzureOpenAILLM.from_arg_string(
        model="gpt-4o", arg_string="azure_endpoint=test,api_version=test,api_key=API_KEY"
    )
    mock_create = mocker.patch.object(llm.openai_client.chat.completions, "create")
    mock_create.side_effect = _bad_request("The response was filtered due to the prompt triggering content_filter.")

    with pytest.raises(ContentFilterException):
        llm.invoke([LLMMessage(role=LLMRole.USER, content="Hello")])


def test_openai_filtered_answer_raises_content_filter(mocker: MockerFixture):
    import pytest

    from lmbench.models.exceptions import ContentFilterException

    os.environ["API_KEY"] = "test_api_key"
    llm = OpenAILLM.from_arg_string(model="gpt-4o", arg_string="base_url=test,api_key=API_KEY")
    mock_create = mocker.patch.object(llm.openai_client.chat.completions, "create")
    choice = MagicMock()
    choice.finish_reason = "content_filter"
    choice.message.content = ""
    mock_create.return_value = MagicMock(choices=[choice])

    with pytest.raises(ContentFilterException):
        llm.invoke([LLMMessage(role=LLMRole.USER, content="Hello")])


def test_openai_context_size_error_still_maps_to_shrinking(mocker: MockerFixture):
    import pytest

    from lmbench.models.exceptions import ContextSizeException

    os.environ["API_KEY"] = "test_api_key"
    llm = OpenAILLM.from_arg_string(model="gpt-4o", arg_string="base_url=test,api_key=API_KEY")
    mock_create = mocker.patch.object(llm.openai_client.chat.completions, "create")
    mock_create.side_effect = _bad_request("maximum context length is 8192 tokens, you requested 9000 tokens")

    with pytest.raises(ContextSizeException):
        llm.invoke([LLMMessage(role=LLMRole.USER, content="Hello")])


def test_content_filter_rejection_skips_the_tenacity_retry(mocker: MockerFixture, monkeypatch):
    import pytest

    from lmbench.models.abstract import LLM
    from lmbench.models.exceptions import ContentFilterException

    monkeypatch.undo()  # restore the retrying invoke the autouse fixture unwraps
    assert hasattr(LLM.invoke, "retry")
    monkeypatch.setenv("API_KEY", "test_api_key")
    llm = OpenAILLM.from_arg_string(model="gpt-4o", arg_string="base_url=test,api_key=API_KEY")
    mock_create = mocker.patch.object(llm.openai_client.chat.completions, "create")
    mock_create.side_effect = _bad_request("ResponsibleAIPolicyViolation: content management policy")

    with pytest.raises(ContentFilterException):
        llm.invoke([LLMMessage(role=LLMRole.USER, content="Hello")])
    assert mock_create.call_count == 1
