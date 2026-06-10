import os

from pytest_mock import MockerFixture

from lmbench.config.config import TOKENIZER_FOLDER
from lmbench.models.data_models import LLMMessage, LLMRole
from lmbench.models.openai import AzureOpenAILLM
from lmbench.tokenizer.huggingface import HuggingfaceTokenizer
from lmbench.tokenizer.simple import SimpleTokenizer


def test_simple_tokenizer():
    os.environ["API_KEY"] = "test_api_key"
    llm = AzureOpenAILLM.from_arg_string(
        model="gpt-4o", arg_string="azure_endpoint=test,api_version=test,api_key=API_KEY"
    )
    s = SimpleTokenizer(llm=llm)
    assert (
        s.num_tokens([LLMMessage(role=LLMRole.USER, content="Das ist ein einfacher Test")])
        == (5 * s.WORD_TO_TOKEN_RATIO) + s.TEMPLATE_OVERHEAD
    ), "Token count should match expected value plus template overhead"


def test_huggingface_tokenizer(mocker: MockerFixture):
    llm = AzureOpenAILLM.from_arg_string(
        model="llama-3.1-70b-instruct", arg_string="azure_endpoint=test,api_version=test,api_key=API_KEY"
    )
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("transformers.AutoTokenizer.from_pretrained")
    hft = HuggingfaceTokenizer(llm=llm)
    assert hft.tokenizer_name == "llama-3-1-70b-instruct", "Tokenizer name should match expected value."

    def new_path_exists(path):
        return path == f"{TOKENIZER_FOLDER}/llama-3-1"

    mocker.patch("os.path.exists", side_effect=new_path_exists)
    assert hft._local_tokenizer_name("llama-3-1-70b-instruct-8bit-ablated-non-commercial-blah") == "llama-3-1", (
        "Tokenizer name should match class of tokenizer."
    )
    assert hft._local_tokenizer_name("llama-3-1") == "llama-3-1", (
        "Exact tokenizer name match should return the same name."
    )
    assert hft._local_tokenizer_name("llama-3-2") is None, "Tokenizer that cannot be found should return None."
