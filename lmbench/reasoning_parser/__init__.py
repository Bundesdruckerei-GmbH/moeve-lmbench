"""Module containing ReasoningParsers."""

from lmbench.reasoning_parser.abstract import ReasoningParser
from lmbench.reasoning_parser.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

REASONING_PARSER_LIST: list[type[ReasoningParser]] = [DeepSeekR1ReasoningParser]
REASONING_PARSER_MAP: dict[str, type[ReasoningParser]] = {
    reason_parser.name: reason_parser for reason_parser in REASONING_PARSER_LIST
}  # type: ignore
