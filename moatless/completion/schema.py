import json
import logging
from collections.abc import Iterable
from typing import (
    ClassVar,
    List,
    Literal,
    Optional,
    TypeVar,
    TypedDict,
    Union,
)

from docstring_parser import parse
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Define a type variable as a replacement for Self in Python 3.10
T = TypeVar("T")


class ChatCompletionCachedContent(TypedDict):
    type: Literal["ephemeral"]


class ChatCompletionToolParamFunctionChunk(TypedDict, total=False):
    name: str
    description: str
    parameters: dict


class ChatCompletionToolParam(TypedDict, total=False):
    type: Union[Literal["function"], str]
    function: ChatCompletionToolParamFunctionChunk
    cache_control: Optional[ChatCompletionCachedContent]


class ChatCompletionTextObject(TypedDict):
    type: Literal["text"]
    text: str
    cache_control: Optional[ChatCompletionCachedContent]


class ChatCompletionThinkingObject(TypedDict):
    type: Literal["thinking"]
    thinking: Optional[str]
    signature: Optional[str]
    data: Optional[str]


class ChatCompletionImageUrlObject(TypedDict, total=False):
    url: str
    detail: str


class ChatCompletionImageObject(TypedDict):
    type: Literal["image_url"]
    image_url: Union[str, ChatCompletionImageUrlObject]


MessageContentListBlock = Union[ChatCompletionTextObject, ChatCompletionImageObject]

MessageContent = Union[
    str,
    Iterable[MessageContentListBlock],
]

ValidUserMessageContentTypes = [
    "text",
    "image_url",
    "input_audio",
]  # used for validating user messages. Prevent users from accidentally sending anthropic messages.


class ChatCompletionUserMessage(TypedDict):
    role: Literal["user"]
    content: MessageContent
    cache_control: Optional[ChatCompletionCachedContent]


class ChatCompletionToolCallFunctionChunk(TypedDict, total=False):
    name: Optional[str]
    arguments: str


class ChatCompletionAssistantToolCall(TypedDict):
    id: Optional[str]
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk


class ChatCompletionMessage(TypedDict, total=False):
    role: Literal["assistant", "user", "tool"]
    content: Optional[MessageContent]


class ChatCompletionAssistantMessage(ChatCompletionMessage, total=False):
    name: Optional[str]
    content: Optional[MessageContent]
    tool_calls: Optional[list[ChatCompletionAssistantToolCall]]
    function_call: Optional[ChatCompletionToolCallFunctionChunk]
    cache_control: Optional[ChatCompletionCachedContent]


class ChatCompletionToolMessage(ChatCompletionMessage):
    tool_call_id: str


class ChatCompletionSystemMessage(TypedDict, total=False):
    role: Literal["system"]
    content: Union[str, list]
    name: str
    cache_control: Optional[ChatCompletionCachedContent]


AllMessageValues = Union[
    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
    ChatCompletionSystemMessage,
]


class NameDescriptor:
    def __get__(self, obj, cls=None) -> str:
        if cls is None:
            return "Unknown"
        if hasattr(cls, "model_config") and "title" in cls.model_config:
            return cls.model_config["title"]
        return cls.__name__


class ResponseSchema(BaseModel):
    name: ClassVar[NameDescriptor] = NameDescriptor()

    @classmethod
    def description(cls):
        """
        Return the description of the schema.
        First tries to get from docstring, then falls back to schema description.
        """
        # Direct docstring extraction
        if cls.__doc__:
            # Clean the docstring by removing leading/trailing whitespace and splitting by lines
            doc_lines = [line.strip() for line in cls.__doc__.strip().split("\n") if line.strip()]
            if doc_lines:
                # Return the first non-empty line as the short description
                return doc_lines[0]

        # Fall back to schema description if docstring extraction fails
        return cls.model_json_schema().get("description", "")

    @classmethod
    def tool_schema(cls, thoughts_in_action: bool = False) -> ChatCompletionToolParam:
        return cls.openai_schema(thoughts_in_action=thoughts_in_action)

    @classmethod
    def openai_schema(cls, thoughts_in_action: bool = False) -> ChatCompletionToolParam:
        """
        Return the schema in the format of OpenAI's schema as jsonschema
        """
        schema = cls.model_json_schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v
            for k, v in schema.items()
            if k not in ("title", "description") and (thoughts_in_action or k != "thoughts")
        }

        if not thoughts_in_action and parameters["properties"].get("thoughts"):
            del parameters["properties"]["thoughts"]

        def remove_defaults(obj: dict) -> None:
            """Recursively remove default fields from a schema object"""
            if isinstance(obj, dict):
                if "default" in obj:
                    del obj["default"]
                # Recurse into nested properties
                for value in obj.values():
                    remove_defaults(value)
            elif isinstance(obj, list):
                for item in obj:
                    remove_defaults(item)

        def resolve_refs(obj: dict, defs: dict) -> dict:
            """Recursively resolve $ref references in the schema"""
            if not isinstance(obj, dict):
                return obj

            result = {}
            for k, v in obj.items():
                if k == "items" and isinstance(v, dict) and "$ref" in v:
                    # Handle array items that use $ref
                    ref_path = v["$ref"]
                    if isinstance(ref_path, str) and ref_path.startswith("#/$defs/"):
                        ref_name = ref_path.split("/")[-1]
                        if ref_name in defs:
                            # Create a new dict with resolved reference
                            resolved = defs[ref_name].copy()
                            # Copy over any other properties from the original item object
                            for k2, v2 in v.items():
                                if k2 != "$ref":
                                    resolved[k2] = v2
                            result[k] = resolve_refs(resolved, defs)  # Recursively resolve any nested refs
                            continue
                elif k == "$ref":
                    ref_path = v
                    if isinstance(ref_path, str) and ref_path.startswith("#/$defs/"):
                        ref_name = ref_path.split("/")[-1]
                        if ref_name in defs:
                            # Create a new dict with all properties except $ref
                            resolved = {k2: v2 for k2, v2 in obj.items() if k2 != "$ref"}
                            # Merge with the referenced definition
                            referenced = defs[ref_name].copy()
                            referenced.update(resolved)
                            return resolve_refs(referenced, defs)
                elif k == "allOf" and isinstance(v, list):
                    # Merge all objects in allOf into a single object
                    merged = {}
                    for item in v:
                        resolved_item = resolve_refs(item, defs)
                        merged.update(resolved_item)
                    # Copy over any other properties from the parent object
                    for other_k, other_v in obj.items():
                        if other_k != "allOf":
                            merged[other_k] = other_v
                    return merged

                # Recursively resolve nested objects/arrays
                if isinstance(v, dict):
                    result[k] = resolve_refs(v, defs)
                elif isinstance(v, list):
                    result[k] = [resolve_refs(item, defs) if isinstance(item, dict) else item for item in v]
                else:
                    result[k] = v

            return result

        # Remove default field from all properties recursively
        remove_defaults(parameters)

        # Resolve all $ref references
        if "$defs" in parameters:
            defs = parameters.pop("$defs")
            parameters = resolve_refs(parameters, defs)

        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (description := param.description):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k
            for k, v in parameters["properties"].items()
            if "default" not in v and (thoughts_in_action or k != "thoughts")
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all " f"the required parameters with correct types"
                )
        name = cls.name
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": schema["description"],
                "parameters": parameters,
            },
        }

    @classmethod
    def model_validate_xml(cls, xml_text: str) -> T:
        """Parse XML format into model fields."""
        parsed_input = {}
        # Get fields from the class's schema
        schema = cls.model_json_schema()
        properties = schema.get("properties", {})

        if "thoughts" in properties:
            del properties["thoughts"]

        xml_fields = list(properties.keys())

        for field in xml_fields:
            start_tag = f"<{field}>"
            end_tag = f"</{field}>"
            if start_tag in xml_text and end_tag in xml_text:
                start_idx = xml_text.index(start_tag) + len(start_tag)
                end_idx = xml_text.index(end_tag)
                content = xml_text[start_idx:end_idx]

                # Handle both single-line and multi-line block content
                if content:
                    # If content starts/ends with newlines, preserve the inner content
                    if content.startswith("\n") and content.endswith("\n"):
                        # Remove first and last newline but preserve internal formatting
                        content = content[1:-1].rstrip("\n")
                    parsed_input[field] = content

        return cls.model_validate(parsed_input)

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        **kwarg,
    ) -> T:
        if not json_data:
            raise ValidationError("Message is empty")
        
        json_data = json_data.strip().removeprefix("```json").removesuffix("```").strip()
        # if(not json_data.startswith('<')):
        #     json_data = lambda json_data: (match.group(1).strip() if (match := re.search(r'```json\s*(.*?)\s*```', json_data, re.DOTALL)) else json_data.strip())
        print(f"【DEBUG】validate_json:{json_data}")
        try:
            parsed_data = json.loads(json_data, strict=False)
            

            def unescape_values(obj):
                if isinstance(obj, dict):
                    return {k: unescape_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [unescape_values(v) for v in obj]
                elif isinstance(obj, str) and "\\" in obj:
                    return obj.encode().decode("unicode_escape")
                return obj

            cleaned_data = unescape_values(parsed_data)
            cleaned_json = json.dumps(cleaned_data)
            return super().model_validate_json(cleaned_json, **kwarg)

        except (json.JSONDecodeError, ValidationError):
            # If direct parsing fails, try more aggressive cleanup
            logger.warning("Initial JSON parse failed, attempting alternate cleanup")

            message = json_data

            # Ensure message is a string
            if not isinstance(message, str):
                message_str = message.decode("utf-8") if isinstance(message, (bytes, bytearray)) else str(message)
            else:
                message_str = message

            # Clean control characters
            cleaned_chars = []
            for char in message_str:
                # Only keep printable characters and specific whitespace
                if ord(char) >= 32 or char in "\n\r\t":
                    cleaned_chars.append(char)
            cleaned_message = "".join(cleaned_chars)

            if cleaned_message != message_str:
                logger.info(f"parse_json() Cleaned control chars: {repr(message_str)} -> {repr(cleaned_message)}")
            message_str = cleaned_message

            # Replace None with null
            message_str = message_str.replace(": None", ": null").replace(":None", ":null")

            # Extract JSON and try parsing again
            message, all_jsons = extract_json_from_message(message_str)
            if all_jsons:
                if len(all_jsons) > 1:
                    logger.warning(f"Found multiple JSON objects, using the first one. All found: {all_jsons}")
                message = all_jsons[0]
            else:
                # If no valid JSON was found, check if it's due to invalid escape sequences
                try:
                    json.loads(message_str)
                except json.JSONDecodeError as e:
                    if "Invalid \\escape" in str(e):
                        raise ValueError(
                            f"JSON contains invalid escape sequences. Please properly escape backslashes in your JSON. "
                            f"For example, use '\\\\' instead of '\\' in string values. Error: {e}"
                        )
                    else:
                        raise ValueError(f"Invalid JSON format: {e}")

            # Normalize line endings
            if isinstance(message, str):
                message = message.replace("\r\n", "\n").replace("\r", "\n")

            return super().model_validate_json(message if isinstance(message, str) else json.dumps(message), **kwarg)

    def format_args_for_llm(self) -> str:
        """
        Format the input arguments for LLM completion calls. Override in subclasses for custom formats.
        Default implementation returns JSON format.
        """
        return json.dumps(
            self.model_dump(exclude={"thoughts"} if hasattr(self, "thoughts") else None),
            indent=2,
        )

    @classmethod
    def format_schema_for_llm(cls, thoughts_in_action: bool = False) -> str:
        """
        Format the schema description for LLM completion calls.
        Default implementation returns JSON schema.
        """
        schema = cls.model_json_schema()

        if not thoughts_in_action and schema["properties"].get("thoughts"):
            del schema["properties"]["thoughts"]
            schema["required"] = sorted(
                k
                for k, v in schema["properties"].items()
                if "default" not in v and (thoughts_in_action or k != "thoughts")
            )

        return f"Requires a JSON response with the following schema: {json.dumps(schema, ensure_ascii=False)}"

    @classmethod
    def format_xml_schema(cls, xml_fields: dict[str, str]) -> str:
        """
        Format XML schema description.
        Used by actions that require XML-formatted input.

        Args:
            xml_fields: Dictionary mapping field names to their descriptions
        """
        schema = ["Requires the following XML format:"]

        # Build example XML structure
        example = []
        for field_name, field_desc in xml_fields.items():
            example.append(f"<{field_name}>{field_desc}</{field_name}>")

        return "\n".join(schema + example)

    @classmethod
    def get_few_shot_examples(cls) -> list["FewShotExample"]:
        """
        Returns a list of few-shot examples specific to this action.
        Override this method in subclasses to provide custom examples.
        """
        return []


class FewShotExample(BaseModel):
    user_input: str = Field(..., description="The user's input/question")
    action: ResponseSchema = Field(..., description="The expected response")

    @classmethod
    def create(cls, user_input: str, action: ResponseSchema) -> "FewShotExample":
        return cls(user_input=user_input, action=action)


def extract_json_from_message(message: str) -> tuple[dict | str, list[dict]]:
    """
    Extract JSON from a message, handling both code blocks and raw JSON.
    Returns a tuple of (selected_json_dict, all_found_json_dicts).
    """

    def fix_json_escape_sequences(json_str: str) -> str:
        """Fix common JSON escape sequence issues that occur in LLM responses."""
        import re
        
        # Fix the specific issue: r'{...}' patterns where \' is invalid
        # This happens when LLMs generate regex patterns with single quotes
        # Replace r\' with r' (removing the backslash before single quote)
        json_str = re.sub(r"r\\(['\"])", r"r\1", json_str)
        
        # Fix other common escape issues:
        # Replace unescaped backslashes in string values (but not already escaped ones)
        # This is a more conservative approach that only fixes obvious issues
        
        # Fix single backslashes that aren't part of valid escape sequences
        # Valid JSON escape sequences: \" \\ \/ \b \f \n \r \t \uXXXX
        valid_escapes = r'["\\\/bfnrt]|u[0-9a-fA-F]{4}'
        
        # Find strings in JSON and fix invalid escapes within them
        def fix_string_escapes(match):
            quote = match.group(1)  # Opening quote
            content = match.group(2)  # String content
            closing_quote = match.group(3)  # Closing quote
            
            # Fix backslashes that aren't followed by valid escape sequences
            fixed_content = re.sub(
                r'\\(?!' + valid_escapes + r')',
                r'\\\\',
                content
            )
            
            return quote + fixed_content + closing_quote
        
        # Match JSON strings (handling escaped quotes within strings)
        string_pattern = r'(")([^"\\]*(?:\\.[^"\\]*)*)(")|(\'([^\'\\]*(?:\\.[^\'\\]*)*)\')'
        json_str = re.sub(string_pattern, fix_string_escapes, json_str)
        
        return json_str

    def clean_json_string(json_str: str) -> str:
        # Remove single-line comments and clean control characters
        lines = []
        for line in json_str.split("\n"):
            # Remove everything after // or #
            line = line.split("//")[0].split("#")[0].rstrip()
            # Clean control characters but preserve newlines and spaces
            line = "".join(char for char in line if ord(char) >= 32 or char in "\n\t")
            if line:  # Only add non-empty lines
                lines.append(line)
        return "\n".join(lines)

    all_found_jsons = []

    # First try to find ```json blocks
    try:
        current_pos = 0
        while True:
            start = message.find("```json", current_pos)
            if start == -1:
                break
            start += 7  # Move past ```json
            end = message.find("```", start)
            if end == -1:
                break
            potential_json = clean_json_string(message[start:end].strip())
            potential_json = fix_json_escape_sequences(potential_json)
            try:
                json_dict = json.loads(potential_json)
                # Validate that this is a complete, non-truncated JSON object
                if isinstance(json_dict, dict) and all(isinstance(k, str) for k in json_dict.keys()):
                    all_found_jsons.append(json_dict)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON block after escape fix: {e}")
            current_pos = end + 3

        if all_found_jsons:
            # Return the most complete JSON object (one with the most fields)
            return max(all_found_jsons, key=lambda x: len(x)), all_found_jsons
    except Exception as e:
        logger.warning(f"Failed to extract JSON from code blocks: {e}")

    # If no ```json blocks found or they failed, try to find raw JSON objects
    try:
        current_pos = 0
        while True:
            start = message.find("{", current_pos)
            if start == -1:
                break
            # Try to parse JSON starting from each { found
            for end in range(len(message), start, -1):
                try:
                    potential_json = clean_json_string(message[start:end])
                    potential_json = fix_json_escape_sequences(potential_json)
                    json_dict = json.loads(potential_json)
                    # Validate that this is a complete, non-truncated JSON object
                    if isinstance(json_dict, dict) and all(isinstance(k, str) for k in json_dict.keys()):
                        all_found_jsons.append(json_dict)
                    break
                except json.JSONDecodeError:
                    continue
            if not all_found_jsons:  # If no valid JSON found, move past this {
                current_pos = start + 1
            else:
                current_pos = end

        if all_found_jsons:
            # Return the most complete JSON object (one with the most fields)
            return max(all_found_jsons, key=lambda x: len(x)), all_found_jsons
    except Exception as e:
        logger.warning(f"Failed to extract raw JSON objects: {e}")

    return message, all_found_jsons
