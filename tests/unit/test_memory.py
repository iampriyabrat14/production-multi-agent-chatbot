from langchain_core.messages import HumanMessage, AIMessage
from memory.short_term import ShortTermMemory


class TestShortTermMemory:

    def test_add_and_get_messages(self):
        mem = ShortTermMemory(max_messages=10)
        mem.add_message("human", "Hello")
        mem.add_message("ai", "Hi there!")
        messages = mem.get_messages()
        assert len(messages) == 2

    def test_max_messages_enforced(self):
        mem = ShortTermMemory(max_messages=3)
        for i in range(5):
            mem.add_message("human", f"Message {i}")
        # Should only keep last 3
        assert mem.count() == 3

    def test_get_last_n(self):
        mem = ShortTermMemory(max_messages=10)
        for i in range(6):
            mem.add_message("human", f"msg {i}")
        last_3 = mem.get_last_n(3)
        assert len(last_3) == 3
        assert "msg 5" in last_3[-1].content

    def test_clear_resets_memory(self):
        mem = ShortTermMemory(max_messages=10)
        mem.add_message("human", "Test message")
        mem.clear()
        assert mem.count() == 0

    def test_message_types_correct(self):
        mem = ShortTermMemory(max_messages=10)
        mem.add_message("human", "Hello")
        mem.add_message("ai", "Hi")
        messages = mem.get_messages()
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_empty_memory_returns_empty_list(self):
        mem = ShortTermMemory(max_messages=10)
        assert mem.get_messages() == []
        assert mem.count() == 0

    def test_oldest_messages_dropped_first(self):
        mem = ShortTermMemory(max_messages=2)
        mem.add_message("human", "first")
        mem.add_message("human", "second")
        mem.add_message("human", "third")
        messages = mem.get_messages()
        contents = [m.content for m in messages]
        assert "first" not in contents
        assert "second" in contents
        assert "third" in contents
