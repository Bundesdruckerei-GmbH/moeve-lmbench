from pytest_httpserver import HTTPServer

from lmbench.models.data_models import LLMMessage, LLMRole
from lmbench.models.ollama import OllamaLLM


def test_ollama_llm(httpserver: HTTPServer):
    model = "dummy"
    request = "some LLM request"
    answer = "the answer form the llm"

    ollama = OllamaLLM(host=httpserver.url_for("/"), model=model)

    httpserver.expect_ordered_request(
        "/api/tags",
    ).respond_with_json(
        {
            "models": [],
        }
    )

    httpserver.expect_ordered_request(
        "/api/pull",
        method="POST",
        json={
            "model": "dummy",
            "insecure": False,
            "stream": False,
        },
    ).respond_with_json(
        {
            "status": "success",
        }
    )

    expected_message = {
        "role": "user",
        "content": request,
    }

    httpserver.expect_ordered_request(
        "/api/chat",
        method="POST",
        json={
            "model": model,
            "messages": [expected_message],
            "tools": [],
            "stream": False,
            "think": False,
            "options": {"num_ctx": 4096, "num_predict": 2048},
        },
    ).respond_with_json(
        {
            "model": model,
            "message": {
                "role": "assistant",
                "content": answer,
            },
        }
    )

    ollama_answer = ollama.invoke([LLMMessage(role=LLMRole.USER, content=request)])
    assert ollama_answer == answer, f"Expected {answer}, but got {ollama_answer}"
