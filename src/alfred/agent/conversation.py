from alfred.llm.base import Message, MessageRole


class Conversation:
    def __init__(self, system_prompt: str):
        self.messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt)
        ]

    def add_user_message(self, content: str):
        self.messages.append(Message(role=MessageRole.USER, content=content))

    def add_assistant_message(self, content: str):
        self.messages.append(Message(role=MessageRole.ASSISTANT, content=content))

    def remove_last_message(self):
        # prevent removing the first message which system prompt
        if len(self.messages) > 1:
            self.messages = self.messages[:-1]

    def get_messages(self):
        return list(self.messages)
