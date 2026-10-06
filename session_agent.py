from agents import SQLiteSession

session = SQLiteSession(session_id="user_chat_thread_001", db_path="agent_memory.db")

if __name__ == "__main__":
    print("session", session)
