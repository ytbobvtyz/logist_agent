#!/usr/bin/env python3
"""
Test script to verify that all errors are fixed.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    modules_to_test = [
        ("utils.config", "settings"),
        ("utils.async_helpers", "start_background_loop"),
        ("core.agent", "LogsitAgent"),
        ("core.conversation_manager", "ConversationManager"),
        ("services.mcp_orchestrator", "MCPOrchestrator"),
        ("app.components.sidebar", "SidebarComponent"),
        ("app.components.chat", "ChatComponent"),
        ("app.handlers.message_handler", "MessageHandler"),
    ]
    
    all_ok = True
    for module_name, item_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[item_name])
            getattr(module, item_name)
            print(f"  ✅ {module_name}.{item_name}")
        except Exception as e:
            print(f"  ❌ {module_name}.{item_name}: {e}")
            all_ok = False
    
    return all_ok

def test_agent_initialization():
    """Test that agent initializes correctly with all attributes."""
    print("\nTesting agent initialization...")
    
    try:
        from utils.async_helpers import start_background_loop, stop_background_loop
        from core.agent import LogsitAgent
        
        print("1. Starting background loop...")
        start_background_loop()
        
        print("2. Creating agent...")
        agent = LogsitAgent()
        
        print("3. Checking required attributes...")
        required_attrs = [
            'conversation_manager',
            'mcp_orchestrator',
            'client',
            'state',
            'rag_service',
            'summarizer',
            'task_state_manager',
            'config',
            'model'
        ]
        
        all_ok = True
        for attr in required_attrs:
            if hasattr(agent, attr):
                value = getattr(agent, attr)
                print(f"  ✅ {attr}: exists ({type(value).__name__})")
            else:
                print(f"  ❌ {attr}: MISSING")
                all_ok = False
        
        if all_ok:
            print("✅ All attributes present!")
        else:
            print("❌ Missing some attributes")
        
        print("\n4. Testing basic functionality...")
        # Test conversation creation
        conversation = agent.conversation_manager.create_conversation()
        print(f"  ✅ Created conversation: {conversation.id}")
        
        # Test getting all conversations
        conversations = agent.get_all_conversations()
        print(f"  ✅ Got {len(conversations)} conversations")
        
        # Test model switching
        success = agent.switch_model("deepseek/deepseek-chat")
        print(f"  ✅ Model switching: {success}")
        
        # Clean up
        stop_background_loop()
        
        return all_ok
        
    except Exception as e:
        print(f"❌ Agent initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_message_handler():
    """Test that message handler can be created."""
    print("\nTesting message handler...")
    
    try:
        from utils.async_helpers import start_background_loop, stop_background_loop
        from core.agent import LogsitAgent
        from app.components.sidebar import SidebarComponent
        from app.components.chat import ChatComponent
        from app.handlers.message_handler import MessageHandler
        
        print("1. Starting background loop...")
        start_background_loop()
        
        print("2. Creating dependencies...")
        agent = LogsitAgent()
        sidebar = SidebarComponent()
        chat = ChatComponent()
        
        print("3. Creating message handler...")
        handler = MessageHandler(agent, sidebar, chat)
        
        print("  ✅ MessageHandler created successfully")
        print(f"     - Agent: {type(handler.agent).__name__}")
        print(f"     - Sidebar: {type(handler.sidebar).__name__}")
        print(f"     - Chat: {type(handler.chat).__name__}")
        
        # Clean up
        stop_background_loop()
        
        return True
        
    except Exception as e:
        print(f"❌ Message handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("Logsit Agent - Post-Fix Verification Test")
    print("=" * 70)
    
    tests = [
        ("Module Imports", test_imports),
        ("Agent Initialization", test_agent_initialization),
        ("Message Handler", test_message_handler),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL ERRORS ARE FIXED! Application should work correctly.")
        print("\nTo run the application:")
        print("  cd /home/ytbob/projects/logsit_agent")
        print("  source venv/bin/activate")
        print("  ./run_logsit.sh  # или python app/main.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. See errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())