import unittest

from alfred.agent.conversation import Conversation
from alfred.llm.base import MessageRole


class ConversationTestCase(unittest.TestCase):
    def setUp(self):
        self.conversation = Conversation(system_prompt="your are an assistant.")

    def test_ctor_initialize_with_system_prompt(self):
        conversation_messages = self.conversation.get_messages()
        self.assertEqual(len(conversation_messages), 1)
        self.assertEqual(conversation_messages[0].role, MessageRole.SYSTEM)
        self.assertEqual(conversation_messages[0].content, "your are an assistant.")

    def test_add_user_message(self):
        self.conversation.add_user_message("what is the capital of France ?")
        conversation_messages = self.conversation.get_messages()
        self.assertEqual(len(conversation_messages), 2)
        self.assertEqual(conversation_messages[1].role, MessageRole.USER)
        self.assertEqual(
            conversation_messages[1].content, "what is the capital of France ?"
        )

    def test_add_assistant_message(self):
        self.conversation.add_assistant_message("Please ask a question")
        conversation_messages = self.conversation.get_messages()
        self.assertEqual(len(conversation_messages), 2)
        self.assertEqual(conversation_messages[1].role, MessageRole.ASSISTANT)
        self.assertEqual(conversation_messages[1].content, "Please ask a question")

    def test_remove_last_message(self):
        self.conversation.add_user_message("what is the capital of France ?")
        self.assertEqual(len(self.conversation.get_messages()), 2)
        self.conversation.remove_last_message()
        conversation_messages = self.conversation.get_messages()
        self.assertEqual(len(conversation_messages), 1)
        self.assertEqual(conversation_messages[0].role, MessageRole.SYSTEM)

    def test_remove_last_message_dont_delete_system_prompt(self):
        self.conversation.remove_last_message()
        self.conversation.remove_last_message()
        conversation_messages = self.conversation.get_messages()
        self.assertEqual(len(conversation_messages), 1)
        self.assertEqual(conversation_messages[0].role, MessageRole.SYSTEM)

    def test_get_messages_should_return_new_copy(self):
        self.assertFalse(
            self.conversation.get_messages() is self.conversation.get_messages()
        )
